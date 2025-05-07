
import os
import pandas as pd
import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Load dataset
df = pd.read_csv("data/violation_training_dataset.csv")
df["label"] = df["violation_label"].apply(lambda x: 0 if str(x).strip() == "no_violation" else 1)

dataset = Dataset.from_pandas(df[["text", "label"]].dropna())

# Tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# TrainingArguments
training_args = TrainingArguments(
    output_dir="outputs/violation_classifier_beast",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# Save model and metrics
trainer.save_model("models/violation_classifier_beast")

eval_metrics = trainer.evaluate()
with open("outputs/violation_classifier_beast_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("✅ Beast training complete. Metrics and model saved.")
