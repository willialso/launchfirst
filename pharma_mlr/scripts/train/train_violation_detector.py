import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import os

# Configuration (add these if missing)
MODEL_PATH = "/app/models/scibert_domain_finetuned"  # Ensure this path is correct
DATA_PATH = "/app/data/violation_dataset.csv"
OUTPUT_DIR = "/app/models/violation_detector"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load & Clean Data ---
df = pd.read_csv(DATA_PATH)

# Handle missing text (NaN/N/A) and ensure strings
df["section_text"] = df["section_text"].fillna("[NO_TEXT]").astype(str)

# Fix: Use 'CFR_Rule_Violation' (not 'violation_label') for labels
df["label"] = df["CFR_Rule_Violation"].apply(
    lambda x: 0 if x == "N/A" else 1  # 'N/A' = no_violation (0), else violation (1)
)

# --- 2. Create Dataset ---
dataset = Dataset.from_dict({
    "text": df["section_text"].tolist(),  # Already converted to str
    "label": df["label"].tolist(),
})

# Split into train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)  # Add seed for reproducibility

# --- 3. Initialize Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,  # Binary classification
    id2label={0: "no_violation", 1: "violation"},
)

# --- 4. Tokenize Dataset ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",  # Pad to max_length
        truncation=True,       # Truncate to max_length
        max_length=512,        # BERT-style models prefer 512
    )

dataset = dataset.map(tokenize_function, batched=True)

# --- 5. Training Setup ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  # Added for eval
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir=f"{OUTPUT_DIR}/logs",  # Optional: Track logs
    logging_steps=50,                  # Optional: Log every 50 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# --- 6. Train & Save ---
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to {OUTPUT_DIR}")
