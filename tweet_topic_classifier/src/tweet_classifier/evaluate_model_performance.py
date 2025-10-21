"""
Compares the model’s predictions with the manually labelled sample, write summary
metrics to CSV, and export per-topic confusion matrices into a PDF.
"""

import sys
import os
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

log_file = open(os.path.join(LOGS_DIR, "evaluate_model_performance.log"), "a")
error_log_file = open(os.path.join(LOGS_DIR, "evaluate_model_performance_error.log"), "a")

sys.stdout = log_file
sys.stderr = error_log_file

print("Evaluating model performance")
sys.stdout.flush()

df = pd.read_csv(os.path.join(PROCESSED_DIR, "sampled_tweets_by_topic.csv"))

topic_metrics = []
topic_names = []

for column in df.columns:
    if column.endswith("_human"):
        base = column.replace("_human", "")
        bert_column = f"{base}_bert"
        if bert_column in df.columns:
            human_labels = df[column].values
            bert_labels = df[bert_column].values
            precision = precision_score(human_labels, bert_labels, zero_division=0)
            recall = recall_score(human_labels, bert_labels, zero_division=0)
            f1 = f1_score(human_labels, bert_labels, zero_division=0)
            accuracy = accuracy_score(human_labels, bert_labels)
            support = int(human_labels.sum())
            topic_metrics.append([base, precision, recall, f1, accuracy, support])
            topic_names.append(base)

metrics_df = pd.DataFrame(
    topic_metrics,
    columns=["topic", "precision", "recall", "f1", "accuracy", "support"],
)

human_columns = [col for col in df.columns if col.endswith("_human")]
bert_columns = [col for col in df.columns if col.endswith("_bert")]

human_matrix = df[human_columns].values
bert_matrix = df[bert_columns].values

micro_precision = precision_score(human_matrix, bert_matrix, average="micro", zero_division=0)
micro_recall = recall_score(human_matrix, bert_matrix, average="micro", zero_division=0)
micro_f1 = f1_score(human_matrix, bert_matrix, average="micro", zero_division=0)
subset_accuracy = accuracy_score(human_matrix, bert_matrix)

summary_metrics = {
    "micro_precision": micro_precision,
    "micro_recall": micro_recall,
    "micro_f1": micro_f1,
    "subset_accuracy": subset_accuracy,
    "macro_precision": metrics_df["precision"].mean(),
    "macro_recall": metrics_df["recall"].mean(),
    "macro_f1": metrics_df["f1"].mean(),
}

with PdfPages(os.path.join(REPORTS_DIR, "model_evaluation.pdf")) as pdf:
    plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    plt.title("Model Evaluation Summary", fontsize=16, fontweight="bold")
    y = 0.9
    for key, value in summary_metrics.items():
        plt.text(0.1, y, f"{key}: {value:.3f}", fontsize=12)
        y -= 0.05
    pdf.savefig()
    plt.close()

    for metric in topic_metrics:
        topic = metric[0]
        human_labels = df[f"{topic}_human"].values
        bert_labels = df[f"{topic}_bert"].values
        conf = confusion_matrix(human_labels, bert_labels)
        conf_norm = conf / conf.sum()
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix for {topic}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        pdf.savefig()
        plt.close()

metrics_path = os.path.join(PROCESSED_DIR, "evaluation_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

log_file.close()
error_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print(f"Saved evaluation metrics to {metrics_path}")
