import torch
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import os
from sklearn.model_selection import train_test_split

# Configuration
MODEL_PATH = "/app/models/scibert_domain_finetuned"
DATA_PATH = "/app/data/chunk_classifier_dataset.csv"
OUTPUT_DIR = "/app/models/chunk_classifier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
texts = df['section_text'].tolist()
labels = df['section_type'].tolist()

# Create label mapping
unique_labels = sorted(list(set(labels)))
label_map = {label: i for i, label in enumerate(unique_labels)}

# Prepare dataset
dataset = Dataset.from_dict({
    "text": texts,
    "label": [label_map[l] for l in labels]
})
dataset = dataset.train_test_split(test_size=0.2)

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(unique_labels),
    id2label={i: label for i, label in enumerate(unique_labels)},
    label2id=label_map
)

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to {OUTPUT_DIR}")
