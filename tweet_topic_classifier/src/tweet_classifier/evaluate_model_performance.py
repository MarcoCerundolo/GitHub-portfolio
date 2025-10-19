"""Evaluate performance using human vs. BERT labels with frequency-weighted aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Topic frequencies derived from reports/topic_frequency.pdf (285-tweet sample adjusted
# to the full-population distribution). Values sum to 1.0 and re-normalise at runtime
# if any topic is missing from the evaluation subset.
TOPIC_FREQUENCY_WEIGHTS: Dict[str, float] = {
    "immigration": 0.025441,
    "climate change": 0.042148,
    "renewable energy": 0.028605,
    "traditional energy": 0.016370,
    "inequality": 0.050122,
    "social policy": 0.054932,
    "taxation": 0.020167,
    "labour market": 0.036368,
    "international trade": 0.023795,
    "economics": 0.067463,
    "european union": 0.038815,
    "public health": 0.039912,
    "gender rights": 0.023416,
    "civil rights": 0.045481,
    "political rights": 0.021391,
    "connecting": 0.092229,
    "elections/voting": 0.066408,
    "self-promotion": 0.173952,
    "anti-establishment": 0.132985,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BERT predictions against human-labelled topics and export a PDF report "
            "with frequency-weighted aggregate metrics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/sampled_tweets_by_topic.csv"),
        help="CSV file containing *_human and *_bert label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_evaluation.pdf"),
        help="Path to write the evaluation PDF report.",
    )
    return parser.parse_args()


def _find_topics(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Return (topic, human_col, bert_col) tuples preserving column order."""
    topics: List[Tuple[str, str, str]] = []
    for column in df.columns:
        if not column.endswith("_human"):
            continue
        topic = column[: -len("_human")]
        bert_column = f"{topic}_bert"
        if bert_column not in df.columns:
            raise ValueError(f"Missing BERT predictions column for topic '{topic}'.")
        topics.append((topic, column, bert_column))
    if not topics:
        raise ValueError("No *_human columns found in the provided dataset.")
    return topics


def _format_topic(topic: str) -> str:
    """Produce a nicely formatted topic label for display."""
    return topic.replace("_", " ").title()


def _attach_weights(metrics_df: pd.DataFrame) -> pd.DataFrame:
    weighted = metrics_df.copy()
    weighted["weight"] = weighted["topic"].map(TOPIC_FREQUENCY_WEIGHTS).fillna(0.0)
    weight_sum = weighted["weight"].sum()
    if weight_sum == 0:
        raise ValueError(
            "Topic frequency weights could not be aligned with the evaluation dataset."
        )
    weighted["normalised_weight"] = weighted["weight"] / weight_sum
    return weighted


def compute_metrics(
    df: pd.DataFrame, topics: Iterable[Tuple[str, str, str]]
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]]:
    topics = list(topics)
    human_cols = [human for _, human, _ in topics]
    bert_cols = [bert for _, _, bert in topics]

    y_true = df[human_cols].to_numpy(dtype=int)
    y_pred = df[bert_cols].to_numpy(dtype=int)

    per_topic_records: List[Dict[str, float]] = []
    confusion_matrices: Dict[str, np.ndarray] = {}

    for idx, (topic, human_col, bert_col) in enumerate(topics):
        true_vals = y_true[:, idx]
        pred_vals = y_pred[:, idx]
        per_topic_records.append(
            {
                "topic": topic,
                "precision": precision_score(true_vals, pred_vals, zero_division=0),
                "recall": recall_score(true_vals, pred_vals, zero_division=0),
                "f1": f1_score(true_vals, pred_vals, zero_division=0),
                "accuracy": accuracy_score(true_vals, pred_vals),
                "support": int(true_vals.sum()),
            }
        )
        confusion_matrices[topic] = confusion_matrix(
            true_vals,
            pred_vals,
            labels=[0, 1],
        )

    metrics_df = pd.DataFrame(per_topic_records).sort_values("topic").reset_index(drop=True)
    weighted_df = _attach_weights(metrics_df)

    aggregate_metrics = {
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": metrics_df["precision"].mean(),
        "macro_recall": metrics_df["recall"].mean(),
        "macro_f1": metrics_df["f1"].mean(),
        "weighted_precision": (weighted_df["precision"] * weighted_df["normalised_weight"]).sum(),
        "weighted_recall": (weighted_df["recall"] * weighted_df["normalised_weight"]).sum(),
        "weighted_f1": (weighted_df["f1"] * weighted_df["normalised_weight"]).sum(),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "samples_f1": f1_score(y_true, y_pred, average="samples", zero_division=0),
    }

    return weighted_df, confusion_matrices, aggregate_metrics


def _add_summary_page(
    pdf: PdfPages, metrics_df: pd.DataFrame, aggregate_metrics: Dict[str, float]
) -> None:
    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.2, 3])
    ax_header = fig.add_subplot(gs[0])
    ax_spacer = fig.add_subplot(gs[1])
    ax_table = fig.add_subplot(gs[2])

    for ax in (ax_header, ax_spacer, ax_table):
        ax.axis("off")

    fig.suptitle(
        "BERT Topic Classifier Evaluation (Weighted)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    ax_header.text(
        0.0,
        0.85,
        "Aggregate Performance",
        fontsize=16,
        fontweight="bold",
        transform=ax_header.transAxes,
    )

    summary_items = [
        ("Micro Precision", "micro_precision"),
        ("Micro Recall", "micro_recall"),
        ("Micro F1", "micro_f1"),
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted Precision", "weighted_precision"),
        ("Weighted Recall", "weighted_recall"),
        ("Weighted F1", "weighted_f1"),
        ("Samples F1", "samples_f1"),
        ("Subset Accuracy", "subset_accuracy"),
    ]

    y_offset = 0.62
    for label, key in summary_items:
        value = aggregate_metrics.get(key, np.nan)
        ax_header.text(
            0.02,
            y_offset,
            f"{label}: {value:.3f}",
            fontsize=12,
            transform=ax_header.transAxes,
        )
        y_offset -= 0.07

    display_df = metrics_df.copy()
    display_df["Topic"] = display_df["topic"].apply(_format_topic)
    display_df = display_df[
        ["Topic", "normalised_weight", "precision", "recall", "f1", "accuracy", "support"]
    ]
    display_df.rename(
        columns={
            "normalised_weight": "Weight",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "accuracy": "Accuracy",
            "support": "Positives",
        },
        inplace=True,
    )

    table_data = [
        [
            row["Topic"],
            f"{row['Weight']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1 Score']:.3f}",
            f"{row['Accuracy']:.3f}",
            int(row["Positives"]),
        ]
        for _, row in display_df.iterrows()
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=list(display_df.columns),
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_confusion_matrix_pages(
    pdf: PdfPages, confusion_matrices: Dict[str, np.ndarray]
) -> None:
    topics = sorted(confusion_matrices.keys())

    for topic in topics:
        cm = confusion_matrices[topic]
        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            cbar=False,
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            ax=ax,
        )
        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("Actual label", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        fig.suptitle(
            f"Confusion Matrix – {_format_topic(topic)}",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                pct = (count / total * 100.0) if total else 0.0
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{pct:.1f}%\n({count})",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    sns.set_theme(style="white")

    if not args.input.exists():
        raise FileNotFoundError(f"Could not find input data at {args.input}")

    df = pd.read_csv(args.input)
    topics = _find_topics(df)

    metrics_df, confusion_matrices, aggregate_metrics = compute_metrics(df, topics)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _add_summary_page(pdf, metrics_df, aggregate_metrics)
        _add_confusion_matrix_pages(pdf, confusion_matrices)

    print(f"Saved weighted evaluation report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
