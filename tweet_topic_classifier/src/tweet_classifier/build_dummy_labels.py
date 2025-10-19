"""Create dummy topic columns from the GPT-labelled dataset."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List

import pandas as pd

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
    parser = argparse.ArgumentParser(description="Expand GPT topic predictions into dummy variables.")
    parser.add_argument(
        "--input",
        default=Path("data/processed/labelled_tweets.csv"),
        type=Path,
        help="CSV containing the GPT-labelled tweets.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/processed/gpt_dummy_labels.csv"),
        type=Path,
        help="Destination CSV with dummy topic columns.",
    )
    return parser.parse_args()


def parse_topics(raw_value: str) -> List[str]:
    if not isinstance(raw_value, str):
        return []
    cleaned = raw_value.strip()
    if not cleaned:
        return []
    try:
        loaded = ast.literal_eval(cleaned)
        if isinstance(loaded, list):
            return [str(item).strip().lower() for item in loaded]
    except (ValueError, SyntaxError):
        pass
    return [token.strip().lower() for token in cleaned.split(",") if token.strip()]


def create_dummy_columns(label_strings: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, int]] = []
    for value in label_strings:
        topics = {topic: 0 for topic in TOPICS}
        parsed = parse_topics(value)
        for topic in parsed:
            if topic in topics:
                topics[topic] = 1
        rows.append(topics)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "gpt_finetuned" not in df.columns:
        raise KeyError("Expected 'gpt_finetuned' column with GPT output.")

    dummy_df = create_dummy_columns(df["gpt_finetuned"])
    combined = pd.concat([df, dummy_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
