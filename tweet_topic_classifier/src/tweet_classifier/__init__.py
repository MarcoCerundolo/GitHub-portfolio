"""
Scripts for the tweet topic classification pipeline.

Each module mirrors one step from the original project:
- gpt_label_tweets: query GPT to generate topic labels.
- build_dummy_labels: expand GPT labels into multi-hot columns.
- train_bert_classifier: fine-tune BERT on the labelled data.
- run_inference: apply the trained model to new tweets.
- summarise_predictions: prepare samples and charts of predictions.
"""
