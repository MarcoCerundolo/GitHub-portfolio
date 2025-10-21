"""
Uses the 100,000 tweets with associated dummies to train BERT. 
The model gets trained on 4/5 of the tweets and gets evaluated on 1/5 of the tweets.
From this collection of tweets which get withheld, the model gets evaluated and a confusion matrix is produced for each topic, to show how well the model has learnt the chatgpt classification. Notice that this is not really want we are interested in because the chatgpt classification is noisy.
The code also produces a dataset ‘comparison.csv’ which allows you to browse the classification to see if it makes sense. Later we evaluate the model on a human labelled sample."
"""

import sys
import os
import pandas as pd

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import io
from sklearn.metrics import precision_score, recall_score, f1_score
import time

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "train_bert_classifier.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "train_bert_classifier_error.log"), "a")

sys.stdout = log_file
sys.stderr = error_log_file

print("This will be logged in train_bert_classifier.log")
sys.stdout.flush()

df = pd.read_csv(os.path.join(PROCESSED_DIR, "gpt_dummy_labels_100000.csv"))

df = df.drop([df.columns[0], "gpt_finetuned"], axis=1)

df = df.rename(columns={"text_translate": "tweet"})

df["labels"] = df[
    [
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
].values.tolist()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_labels = list(train_df["labels"])
test_labels = list(test_df["labels"])

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

train_encodings = tokenizer(list(train_df["tweet"].values), truncation=False, padding=False)
test_encodings = tokenizer(list(test_df["tweet"].values), truncation=False, padding=False)

max_length = max([len(seq) for seq in train_encodings["input_ids"]] + [len(seq) for seq in test_encodings["input_ids"]])

train_encodings = tokenizer(list(train_df["tweet"].values), truncation=True, padding="max_length", max_length=max_length)
test_encodings = tokenizer(list(test_df["tweet"].values), truncation=True, padding="max_length", max_length=max_length)


def convert_to_tf_dataset(encodings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset


train_dataset = convert_to_tf_dataset(train_encodings, train_labels)
test_dataset = convert_to_tf_dataset(test_encodings, test_labels)

train_dataset = train_dataset.shuffle(100).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=19)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric1 = tf.keras.metrics.BinaryAccuracy("accuracy")
metric2 = tf.keras.metrics.Recall()
metric3 = tf.keras.metrics.Precision()

model.compile(optimizer=optimizer, loss=loss, metrics=[metric1, metric2, metric3])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(train_dataset, epochs=25, validation_data=test_dataset, callbacks=[early_stopping])

results = model.evaluate(test_dataset)

print(f"loss: {results[0]}")
print(f"accuracy: {results[1]}")
print(f"recall: {results[2]}")
print(f"precision: {results[3]}")

model_path = os.path.join(MODELS_DIR, "model_100000")
tokenizer_path = os.path.join(MODELS_DIR, "tokenizer_100000")

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

test_pred_logit = model.predict(test_dataset).logits
test_pred_prob = tf.nn.sigmoid(test_pred_logit).numpy()
test_pred_labels = (test_pred_prob > 0.5).astype(int)

actual_labels = np.array([label for label in test_labels])

comparison_df = pd.DataFrame(
    {
        "tweet": list(test_df["tweet"].values),
        "actual_labels": list(actual_labels),
        "predicted_labels": list(test_pred_labels),
        "predicted_probs": list(test_pred_prob),
    }
)

topic_names = [
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

actual_df = pd.DataFrame(comparison_df["actual_labels"].tolist(), columns=[f"{topic}_actual" for topic in topic_names])
predicted_df = pd.DataFrame(comparison_df["predicted_labels"].tolist(), columns=[f"{topic}_predicted" for topic in topic_names])

comparison_df = pd.concat([comparison_df, actual_df, predicted_df], axis=1)
comparison_df = comparison_df.drop(columns=["actual_labels", "predicted_labels"])

comparison_path = os.path.join(PROCESSED_DIR, "comparison_100000.csv")
comparison_df.to_csv(comparison_path, index=False)

with PdfPages(os.path.join(REPORTS_DIR, "confusion_matrices_100000.pdf")) as pdf:
    for topic in topic_names:
        actual = comparison_df[f"{topic}_actual"]
        predicted = comparison_df[f"{topic}_predicted"]
        conf_matrix = confusion_matrix(actual, predicted)
        conf_matrix_normalized = conf_matrix / conf_matrix.sum()
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix for {topic}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        pdf.savefig()
        plt.close()

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Saved model to {model_path} and tokenizer to {tokenizer_path}")
