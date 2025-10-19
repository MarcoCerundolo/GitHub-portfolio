"""Run inference on a batch of tweets using the fine-tuned classifier."""

from __future__ import annotations

import argparse
import pandas as pd
import tensorflow as tf
from pathlib import Path
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
    parser = argparse.ArgumentParser(description="Generate model predictions for tweets.")
    parser.add_argument(
        "--input",
        default=Path("data/raw/sample_tweets.csv"),
        type=Path,
        help="CSV file with tweets to classify.",
    )
    parser.add_argument(
        "--model-dir",
        default=Path("models/bert_topic_classifier"),
        type=Path,
        help="Directory containing the saved model and tokenizer.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/processed/inference_results.csv"),
        type=Path,
        help="Where to store the predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()


def encode_texts(tokenizer: BertTokenizer, texts, max_length: int = 160):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )
    dataset = tf.data.Dataset.from_tensor_slices(dict(encodings))
    return dataset.batch(32).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    if "text_translate" not in df.columns:
        raise KeyError("Expected 'text_translate' column in input data.")

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = TFBertForSequenceClassification.from_pretrained(args.model_dir)

    texts = df["text_translate"].astype(str).tolist()
    dataset = encode_texts(tokenizer, texts)

    logits = model.predict(dataset).logits
    probabilities = tf.nn.sigmoid(logits).numpy()
    predictions = (probabilities >= 0.5).astype(int)

    df["predicted_labels"] = predictions.tolist()
    df["probabilities"] = probabilities.tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
