"""Fine-tune a BERT model on GPT-labelled tweets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

TOPICS = [
    "immigration",
    "climate change",
    "renewable energy",
    "traditional energy",
    "inequality",
    "social policy",
    "taxation",
    "labour market",
    "international trade",
    "economics",
    "european union",
    "public health",
    "gender rights",
    "civil rights",
    "political rights",
    "connecting",
    "elections/voting",
    "self-promotion",
    "anti-establishment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on GPT-labelled tweets.")
    parser.add_argument(
        "--input",
        default=Path("data/processed/gpt_dummy_labels.csv"),
        type=Path,
        help="CSV with GPT labels and dummy topic columns.",
    )
    parser.add_argument(
        "--model-dir",
        default=Path("models/bert_topic_classifier"),
        type=Path,
        help="Output directory for the saved model and tokenizer.",
    )
    parser.add_argument(
        "--reports-dir",
        default=Path("reports"),
        type=Path,
        help="Directory where evaluation artefacts will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Adam learning rate.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def prepare_labels(df: pd.DataFrame) -> List[List[int]]:
    return df[TOPICS].astype(int).values.tolist()


def build_dataset(tokenizer: BertTokenizer, texts: List[str], labels: List[List[int]] | None, batch_size: int) -> tf.data.Dataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=160,
        return_tensors="tf",
    )

    inputs = {key: tensor for key, tensor in encodings.items()}

    if labels is not None:
        label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, label_tensor))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    configure_logging()

    df = pd.read_csv(args.input)
    if "text_translate" not in df.columns:
        raise KeyError("Expected 'text_translate' column with tweet text.")

    labels = prepare_labels(df)
    texts = df["text_translate"].astype(str).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
    )

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-large-uncased",
        num_labels=len(TOPICS),
    )

    train_dataset = build_dataset(tokenizer, train_texts, train_labels, args.batch_size)
    val_dataset = build_dataset(tokenizer, val_texts, val_labels, args.batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    eval_results = model.evaluate(val_dataset, return_dict=True)
    logging.info("Evaluation: %s", eval_results)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    history_df = pd.DataFrame(history.history)
    history_path = args.reports_dir / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)

    val_probs = tf.nn.sigmoid(model.predict(val_dataset).logits).numpy()
    val_preds = (val_probs >= 0.5).astype(int)

    comparison = pd.DataFrame(
        {
            "tweet": val_texts,
            "true_labels": val_labels,
            "predicted_labels": val_preds.tolist(),
            "probabilities": val_probs.tolist(),
        }
    )
    comparison_path = args.reports_dir / "validation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)


if __name__ == "__main__":
    main()
