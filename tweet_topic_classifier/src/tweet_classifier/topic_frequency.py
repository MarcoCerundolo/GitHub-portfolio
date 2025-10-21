"Counts how many tweets get classified into each topic and how many many tweets don’t get classified into any topic."

import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "summarise_predictions.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "summarise_predictions_error.log"), "a")

sys.stdout = log_file
sys.stderr = error_log_file

print("Summarising predictions")
sys.stdout.flush()


def read_csv_from_local(filename):
    return pd.read_csv(filename, engine="python")


if __name__ == "__main__":
    filename = os.path.join(PROCESSED_DIR, "inference_tweets_100000.csv")
    data = read_csv_from_local(filename)
    print(data.head())

vector_columns = [
    "immigration",
    "climate_change",
    "renewable_energy",
    "traditional_energy",
    "inequality",
    "social_policy",
    "taxation",
    "labour_market",
    "international_trade",
    "economics",
    "eu",
    "public_health",
    "gender_rights",
    "civil_rights",
    "political_rights",
    "connecting",
    "elections",
    "self_promotion",
    "anti_establishment",
]

dummy_variables = [
    "immigration",
    "climate_change",
    "renewable_energy",
    "traditional_energy",
    "inequality",
    "social_policy",
    "taxation",
    "labour_market",
    "international_trade",
    "economics",
    "eu",
    "public_health",
    "gender_rights",
    "civil_rights",
    "political_rights",
    "connecting",
    "elections",
    "self_promotion",
    "anti_establishment",
]

df = data[["tweet_id", "predicted_label"]]

df["predicted_label"] = df["predicted_label"].apply(lambda x: [int(i) for i in x.strip("[]").split(",")])

df[vector_columns] = pd.DataFrame(df["predicted_label"].tolist(), index=df.index)

df = df[
    [
        "tweet_id",
        "immigration",
        "climate_change",
        "renewable_energy",
        "traditional_energy",
        "inequality",
        "social_policy",
        "taxation",
        "labour_market",
        "international_trade",
        "economics",
        "eu",
        "public_health",
        "gender_rights",
        "civil_rights",
        "political_rights",
        "connecting",
        "elections",
        "self_promotion",
        "anti_establishment",
    ]
]

dummy_sums = df.iloc[:, 1:].sum()

df["no_topic_assigned"] = (df.iloc[:, 1:] == 0).all(axis=1)
no_topic_count = df["no_topic_assigned"].sum()

dummy_sums["no_topic"] = no_topic_count

plt.figure(figsize=(10, 6))
dummy_sums.plot(kind="bar", color="blue", edgecolor="black")
plt.title("Number of Tweets within each Topic")
plt.xlabel("Topics")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=90)
plt.tight_layout()

with PdfPages(os.path.join(REPORTS_DIR, "topic_frequency.pdf")) as pdf:
    pdf.savefig()
    plt.close()

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print("Saved topic frequency chart to reports/topic_frequency.pdf")
