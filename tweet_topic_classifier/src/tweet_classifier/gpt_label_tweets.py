"""Generate topic labels for tweets using the OpenAI API.

This script intentionally mirrors the structure of the original project while
cleaning up error handling and configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI, APIError

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

SYSTEM_PROMPT = """You are a political research assistant. Classify the tweet into the allowed topics.
Return a JSON array named 'topics' with only the labels listed below. Use ['no topic'] when none apply.
Allowed topics: {topics}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label tweets with GPT topic predictions.")
    parser.add_argument(
        "--input",
        default=Path("data/raw/sample_tweets.csv"),
        type=Path,
        help="CSV file containing the tweets (must have a 'text_translate' column).",
    )
    parser.add_argument(
        "--output",
        default=Path("data/processed/labelled_tweets.csv"),
        type=Path,
        help="Where to store the GPT-labelled dataset.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat completion model to use.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay in seconds between API calls.")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of retries per tweet on failure.")
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample this many tweets before labelling.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the API key.")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("gpt_label_tweets")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]
    return logger


def normalise_topics(topics: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for topic in topics:
        topic_lower = topic.strip().lower()
        if not topic_lower:
            continue
        if topic_lower == "no topic":
            if cleaned:
                continue
            cleaned.append("no topic")
        elif topic_lower in TOPICS and topic_lower not in seen:
            cleaned.append(topic_lower)
            seen.add(topic_lower)
    if not cleaned:
        cleaned = ["no topic"]
    return cleaned


def extract_topics(message_content: str) -> List[str]:
    text = (message_content or "").strip()
    if not text:
        return ["no topic"]
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "topics" in payload:
            value = payload["topics"]
        else:
            value = payload
        if isinstance(value, list):
            return [str(item) for item in value]
    except json.JSONDecodeError:
        pass
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return [token.strip() for token in text.split(",") if token.strip()]


def classify_tweet(client: OpenAI, text: str, model: str, delay: float, retries: int, logger: logging.Logger) -> List[str]:
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(topics=", ".join(TOPICS))},
                    {"role": "user", "content": text},
                ],
            )
            message = response.choices[0].message.content or ""
            topics = extract_topics(message)
            return normalise_topics(topics)
        except APIError as exc:
            logger.warning("OpenAI error on attempt %s: %s", attempt, exc)
            time.sleep(delay * attempt)
    logger.error("Failed to classify tweet after %s retries", retries)
    return ["no topic"]


def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set the {args.api_key_env} environment variable with your OpenAI API key.")

    client = OpenAI(api_key=api_key)
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    logger = setup_logger(log_path / "gpt_label_tweets.log")

    data = pd.read_csv(args.input)
    if "text_translate" not in data.columns:
        raise KeyError("Input file must contain a 'text_translate' column with the tweet text.")

    if args.sample_size:
        data = data.sample(n=min(args.sample_size, len(data)), random_state=42).reset_index(drop=True)

    labelled = data.copy()
    predictions: List[List[str]] = []

    for idx, text in enumerate(labelled["text_translate"]):
        topics = classify_tweet(client, str(text), args.model, args.sleep, args.max_retries, logger)
        predictions.append(topics)
        if (idx + 1) % 25 == 0:
            logger.info("Processed %s tweets", idx + 1)
        time.sleep(args.sleep)

    labelled["gpt_finetuned"] = [json.dumps(topics) for topics in predictions]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(args.output, index=False)
    logger.info("Saved labelled dataset to %s", args.output)


if __name__ == "__main__":
    main()
