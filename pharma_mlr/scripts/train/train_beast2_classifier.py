import argparse
import logging
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": (preds == labels).mean(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    logger.info("Loading data...")
    df = pd.read_csv(args.csv)
    df = df[df["chunk_text"].notna() & df["label"].notna()]

    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])
    labels = label_encoder.classes_

    logger.info("Loading tokenizer and model...")
    tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")
    model = BertForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", num_labels=len(labels)
    )

    logger.info("Tokenizing dataset...")
    dataset = Dataset.from_pandas(df[["chunk_text", "label_encoded"]])
    dataset = dataset.rename_columns({"chunk_text": "text", "label_encoded": "label"})

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_test = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    logger.info("Initializing trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating model...")
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true = predictions.label_ids

    logger.info("\n" + classification_report(true, preds, target_names=labels))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(true, preds)))

    # Save model + label classes
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    pd.Series(labels).to_csv(f"{args.output_dir}/label_classes.csv", index=False)

if __name__ == "__main__":
    main()
