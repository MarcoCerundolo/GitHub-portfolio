"Use the trained BERT model to classify the full sample of tweets."

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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "run_inference.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "run_inference_error.log"), "a")

sys.stdout = log_file
sys.stderr = error_log_file

print("This will be logged in run_inference.log")
sys.stdout.flush()

tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_physical_devices("GPU")
print("Num GPUs Available: ", len(gpus))

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

if len(tf.config.list_physical_devices("GPU")) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
    print("TensorFlow is using the GPU.")
else:
    print("No GPU found. TensorFlow will use the CPU.")


def read_csv_from_local(filename):
    return pd.read_csv(filename, engine="python")


if __name__ == "__main__":
    filename = os.path.join(RAW_DIR, "df_full_july_tweets.csv")
    df = read_csv_from_local(filename)
    print(df.head())
    sys.stdout.flush()

print(df.columns)

df.rename(columns={"X.1": "tweet_id"}, inplace=True)

df = df[["tweet_id", "text_translate"]]

df["tweet_id"] = df["tweet_id"].astype(int)

tweet_ids = df["tweet_id"].tolist()
texts = df["text_translate"].tolist()

model = TFBertForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "model_100000"))
print("model loaded")
tokenizer = BertTokenizer.from_pretrained(os.path.join(MODELS_DIR, "tokenizer_100000"))
print("tokenizer loaded")


def encode(data):
    encodings = tokenizer(
        list(data["text_translate"].values), truncation=True, padding="max_length", max_length=150
    )
    return encodings


encoded_data = encode(df)

input_ids = encoded_data["input_ids"]
attention_mask = encoded_data["attention_mask"]
token_type_ids = encoded_data["token_type_ids"]

print("data is encoded")
sys.stdout.flush()

dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, token_type_ids))

dataset = dataset.batch(2048)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

print("data is prefetched")
sys.stdout.flush()

predictions = []
for batch in dataset:
    input_ids_batch, attention_mask_batch, token_type_ids_batch = batch
    outputs = model(
        input_ids=input_ids_batch, attention_mask=attention_mask_batch, token_type_ids=token_type_ids_batch
    )
    predictions.append(outputs.logits)

predictions = tf.concat(predictions, axis=0)

probabilities = tf.nn.sigmoid(predictions)

threshold = 0.5
binary_predictions = tf.cast(probabilities > threshold, tf.int32)

binary_predictions_numpy = binary_predictions.numpy()
probabilities_numpy = probabilities.numpy()

labelled_tweets = pd.DataFrame(
    {
        "tweet_id": tweet_ids,
        "text": texts,
        "probabilities": [list(prob) for prob in probabilities_numpy],
        "predicted_label": [list(pred) for pred in binary_predictions_numpy],
    }
)

labelled_tweets.head(5)
sys.stdout.flush()

output_path = os.path.join(PROCESSED_DIR, "inference_tweets_100000.csv")
labelled_tweets.to_csv(output_path, index=False)

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Saved inference results to {output_path}")
