"This script transforms the string output from chatgpt into a set of 19 dummies."

import sys
import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "build_dummy_labels.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "build_dummy_labels_error.log"), "a")

sys.stdout = log_file
sys.stderr = error_log_file

print("Starting dummy label generation")
sys.stdout.flush()


def read_csv_from_local(filename):
    return pd.read_csv(filename, engine="python")


if __name__ == "__main__":
    filename = os.path.join(PROCESSED_DIR, "labelled_tweets_100000.csv")
    df = read_csv_from_local(filename)
    print(df.head())

topics = [
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


def generate_dummies(text):
    dummies = {topic: 0 for topic in topics}
    for topic in topics:
        if topic in text:
            dummies[topic] = 1
    return dummies


dummy_df = df["gpt_finetuned"].apply(generate_dummies).apply(pd.Series)
df = pd.concat([df, dummy_df], axis=1)

output_path = os.path.join(PROCESSED_DIR, "gpt_dummy_labels_100000.csv")
df.to_csv(output_path)

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Saved dummy labels to {output_path}")
