## Tweet Topic Classifier 

## Overview 

This project builds a multi-class topic classifier for Tweets using a weak-supervision approach. I used a large language model (ChatGPT) to automatically label 100,000 tweets across 19 custom-defined topics, then distilled the knowledge into a fine-tuned BERT model. Final performance was evaluated on a human-labeled dataset.

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
- A sample of 285 tweets which have been classified by the model and then human-labeled are included in `data/processed/sampled_tweets_by_topic.csv`.
- If an input dataset is provided then all preprocessing and labeling steps are fully reproducible via the scripts in `src/`.

Source: public API (prior to Musk ownership)
Time period: 2010-2020. 
Sample size: 3.7 million tweets
Preprocessing steps: translated into English

## Labelling Strategy

I defined 19 topic categories and then used ChatGPT with a structured prompt to label 100,000 tweets using these 19 topics (where each tweet can belong to multiple or no topics). The reason for not using human labelling is that 19 are too many topics to label manually, especially since many topics have low frequency.

## Model Architecture

Base model: bert-base-uncased
Model gets trained on 4/5 of the tweets and gets evaluated on 1/5 of the tweets.
To avoid overfitting I use early stopping with a patience of 3 epochs.

## Evaluation Method

For each topic I randomly sampled 15 tweets, from the classified final dataset, creating a dataset of 285 tweets. Note that since a tweet can have multiple topics this mean that the dataset has at least 15 tweets for each topic. I then manually labelled the topics for these tweets. I use this manual dataset to compute the confusion matrix for each topic and calculate a range of key metrics.

## Results & Insights

- The fine-tuned BERT model, as detailed in the [model evaluation report](reports/model_evaluation.pdf), reaches **Micro F1 0.775**, **Macro F1 0.788**, and **Subset Accuracy 0.498** on the 285-tweet human-labeled set, with Micro Precision/Recall at 0.689/0.886.

- Policy-focused topics such as Public Health (F1 0.952), Taxation (0.941), and Labour Market (0.878) exhibit high precision and recall, while broader narratives like Anti-Establishment (F1 0.676) and Civil Rights (0.627) lean heavily on recall and highlight where additional precision-focused examples would help ([model evaluation report](reports/model_evaluation.pdf)).
  
- The [topic frequency overview](reports/topic_frequency.pdf) charts the distribution across all 20 topics (plus a `no_topic` catch-all) over the 3.7M tweet corpus, underscoring the class imbalance with counts on the order of 10^6 tweets for the largest categories.

## Folder Layout

```
tweet_topic_classifier2/
├── data/
│   ├── raw/          # Input CSV files (example: df_full_july_tweets.csv)
│   ├── interim/      # Spare folder for working files
│   └── processed/    # Outputs from each stage of the pipeline
├── logs/             # Plain text logs written by each script
├── models/           # Saved TensorFlow BERT checkpoints
├── reports/          # PDF summaries (topic frequencies, confusion matrices, etc.)
└── src/tweet_classifier/
    ├── gpt_label_tweets.py
    ├── build_dummy_labels.py
    ├── train_bert_classifier.py
    ├── run_inference.py
    ├── topic_frequency.py
    └── evaluate_model_performance.py
```

## How to Run

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # needed for GPT labelling
```

## Running the Pipeline

1. **Label tweets with GPT**
   ```bash
   python3 src/tweet_classifier/gpt_label_tweets.py
   ```
   Reads `data/raw/df_full_july_tweets.csv` and saves `data/processed/labelled_tweets_100000.csv`.

2. **Generate dummy topic columns**
   ```bash
   python3 src/tweet_classifier/build_dummy_labels.py
   ```
   Creates `data/processed/gpt_dummy_labels_100000.csv`.

3. **Train the BERT model**
   ```bash
   python3 src/tweet_classifier/train_bert_classifier.py
   ```
   Saves the fine-tuned model to `models/model_100000` and confusion matrices to `reports/confusion_matrices_100000.pdf`.

4. **Run inference on all tweets**
   ```bash
   python3 src/tweet_classifier/run_inference.py
   ```
   Produces `data/processed/inference_tweets_100000.csv`.

5. **Summarise predictions**
   ```bash
   python3 src/tweet_classifier/topic_frequency.py
   ```
   Creates the topic-frequency bar chart and saves it as `reports/topic_frequency.pdf`.

6. **Evaluate model performance**
   ```bash
   python3 src/tweet_classifier/evaluate_model_performance.py
   ```
   Reads the manually labelled sample (`data/processed/sampled_tweets_by_topic.csv`) and writes an overview to `reports/model_evaluation.pdf`.

Each script writes its own log file under `logs/` so I can check progress when running long jobs.
