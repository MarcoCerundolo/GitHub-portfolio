## Tweet Topic Classifier 

## Overview 

This project builds a multi-class topic classifier for Tweets using a weak-supervision approach. I used a large language model (ChatGPT) to automatically label 100,000 tweets across 20 custom-defined topics, then distilled the knowledge into a fine-tuned BERT model. Final performance was evaluated on a human-labeled dataset.

## Key Features

- LLM-assisted labeling (weak supervision) for large-scale data
- Multi-class classification (20 topics)
- End-to-end preprocessing and modeling pipeline in Python
- BERT fine-tuning with reproducible training scripts
- Manual gold standard evaluation for label quality
- Confusion matrix and per-class performance analysis
- Clean project structure following industry best practices

## Motivation

The aim of the project was to measure group differences between groups in a highly multi-dimensional space such as text. Groups could be populists and non-populists or different families of political parties. These differences can then be measured over time and having the topic of each tweet allows us to see whether the differences are driven by differences within the same topics or driven by differences in topic.

## Dataset

Raw Twitter data used in this project is not included in the repository due to property restrictions.

However:
- A small sample is provided in `data/raw/sample_tweets.csv` to illustrate the format.
- The human-labeled test set (285 tweets) is included in `data/processed/sampled_tweets_by_topic.csv`.
- If an input dataset is provided then all preprocessing and labeling steps are fully reproducible via the scripts in `src/`.

Source: public API (prior to Musk ownership)
Time period: 2010-2020. 
Sample size: 3.7 million tweets
Preprocessing steps: translated into English

## Labelling Strategy

I defined 19 topic categories and then used ChatGPT with a structured prompt to label 100,000 tweets using these 19 topics (where each tweet can belong to multiple or no topics). The reason for not using human labelling is that 19 are too many topics to label manually, especially since many topics have low frequency.

## Model Architecture

Base model: bert-base-uncased
Why BERT / transformer?
Fine-tuning approach (train/val split, hyperparameters)
Handling class imbalance (e.g. class weighting, sampling)

## Evaluation Method

Two-stage evaluation (very impressive):
Internal: BERT performance vs. LLM labels
External: Final evaluation on human-labeled gold set (~200 tweets)
Metrics:
Accuracy
Macro F1 (important for class imbalance)
Per-class precision / recall
Confusion matrix
Qualitative error analysis (optional)

## Results & Insights

The fine-tuned BERT model achieved:

- **Macro F1:** 0.788

![Model Evaluation](reports/model_evaluation)

![Topic Frequency](reports/topic_frequency.pdf)

## Directory Overview

```
tweet_topic_classifier/
├── data/
│   ├── raw/          # Only contains sample of source tweets 
│   ├── interim/   
│   └── processed/    
├── models/           # Saved BERT checkpoints
├── reports/          # Summary PDFs
├── src/tweet_classifier/
│   ├── gpt_label_tweets.py
│   ├── build_dummy_labels.py
│   ├── train_bert_classifier.py
│   ├── run_inference.py
│   ├── summarise_predictions.py
│   └── evaluate_model_performance.py
└── requirements.txt
```

## How to Run

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # needed for GPT labelling
```

### 2. Label Tweets with GPT

```bash
python src/tweet_classifier/gpt_label_tweets.py \
  --input data/raw/raw_tweets.csv \ 
  --output data/processed/labelled_tweets.csv # note: input data not included, default is sample 
```
python3 src/tweet_classifier/gpt_label_tweets.py \
  --input data/raw/sample_tweets.csv \
  --output data/processed/labelled_tweets.csv 

The script stores GPT output in a `gpt_finetuned` column, mirroring the original pipeline.

### 3. Build Dummy Topic Columns

```bash
python3 src/tweet_classifier/build_dummy_labels.py \
  --input data/processed/labelled_tweets.csv \
  --output data/processed/gpt_dummy_labels.csv
```

Each topic becomes a binary column that feeds directly into the BERT model.

### 4. Train the BERT Classifier

```bash
python3 src/tweet_classifier/train_bert_classifier.py \
  --input data/processed/gpt_dummy_labels.csv \
  --model-dir models/bert_topic_classifier \
  --reports-dir reports \
  --epochs 5 \
  --batch-size 16
```

Outputs:
- Saved model/tokenizer under `models/bert_topic_classifier`
- `reports/training_history.csv`
- `reports/validation_comparison.csv`

### 5. Run Inference on New Tweets

```bash
python3 src/tweet_classifier/run_inference.py \
  --input data/raw/new_tweets.csv \
  --model-dir models/bert_topic_classifier \
  --output data/processed/inference_results.csv
```

Predictions include per-topic probabilities plus multi-hot labels.

### 6. Summarise Predictions

```bash
python3 src/tweet_classifier/summarise_predictions.py \
  --input data/processed/inference_results.csv \
  --sample-output data/processed/topic_samples.csv \
  --report reports/topic_frequency.pdf
```

Generates a human-readable sample for QA and a topic distribution chart/table.

### 7. Evaluate Model Performance

```bash
python3 src/tweet_classifier/evaluate_model_performance.py \
  --input data/processed/sampled_tweets_by_topic.csv \
  --output reports/model_evaluation.pdf
```

This script computes aggregate metrics (micro/macro precision, recall, F1, subset accuracy) and per-topic scores, then compiles the results alongside confusion matrices for every topic into a clean PDF.



