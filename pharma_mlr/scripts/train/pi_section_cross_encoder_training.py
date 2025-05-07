#!/usr/bin/env python3
"""
PI Section Cross-Encoder Training

This script fine-tunes a cross-encoder to decide if a promo section’s text matches 
the corresponding PI section’s text. It assumes:
  - Promo files are stored in separate compliant and noncompliant directories.
  - PI files are stored in separate compliant and noncompliant directories.
  - File naming follows:
      Promo: [DRUG]_[compliant|noncompliant]_[number].json
      PI:    [DRUG]_PI_[compliant|noncompliant]_[number].json
  - In a promo, a section contains a non-empty "data_source" field in the format "[PI] - SECTION_NAME"
    (e.g. "[PI] - CLINICAL_STUDIES"). It also has a "data_completeness" field ("complete" or otherwise).
  - The script forms positive pairs by matching a promo with its corresponding PI file (by drug and identifier)
    and negative pairs by pairing a promo with a PI from a different drug (within the same compliance category).
    
The dataset is used to fine-tune a cross-encoder (BERT-based) using the Hugging Face Trainer API.
Each example is constructed as the concatenation of the promo text and the corresponding PI text.
The script then evaluates the model on a hold-out test set and saves evaluation details and the final model.

Usage Example (from docker):
  docker run --rm -v ~/Desktop/pharma_mlr:/app -w /app/scripts/train my-pharma-mlr \
    python pi_section_cross_encoder_training.py \
      --promo_compliant_dir /app/data/case_studies/compliant \
      --promo_noncompliant_dir /app/data/case_studies/non_compliant \
      --pi_compliant_dir /app/data/case_studies/PI_compliant \
      --pi_noncompliant_dir /app/data/case_studies/PI_noncompliant \
      --negative_ratio 1.0 \
      --model_output /app/models/pi_section_matching_model \
      --output_details_csv pi_section_matching_evaluation_details.csv \
      --output_dir /app/outputs \
      --test_size 0.2 \
      --epochs 3 \
      --lr 2e-5 \
      --batch_size 8 \
      --weight_decay 0.01 \
      --seed 42 \
      --max_length 256 \
      --augment True \
      --device cpu
"""

import argparse, csv, glob, json, logging, os, random, sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

####################################
# Data Pairing Functions
####################################
def extract_id_info(filename):
    """
    Extract drug name and identifier from filename.
    Expected formats:
      Promo: [DRUG]_[compliant|noncompliant]_[number].json
      PI:    [DRUG]_PI_[compliant|noncompliant]_[number].json
    Returns (drug, number) with drug in uppercase.
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None, None
    drug = parts[0].upper()
    num = parts[-1].split(".")[0]
    return drug, num

def extract_data_source_field(promo_path):
    """
    From a promo JSON file, search its sections for the first section that has a non-empty
    "data_source" field. Return a tuple (promo_text, referenced_PI_section, completeness_flag).
    Expected data_source format: "[PI] - SECTION_NAME". 
    completeness_flag is 1 if "data_completeness" is "complete", else 0.
    """
    try:
        with open(promo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "sections" in data:
            for sec in data["sections"]:
                ds = sec.get("data_source", "").strip()
                if ds and ds.upper() != "N/A":
                    promo_text = sec.get("text", "").strip()
                    if not promo_text or promo_text.upper() == "N/A":
                        continue
                    dc = sec.get("data_completeness", "").strip().lower()
                    comp_flag = 1 if dc == "complete" else 0
                    if ds.upper().startswith("[PI] -"):
                        ref_section = ds[6:].strip().upper()
                    else:
                        ref_section = ds.upper()
                    return promo_text, ref_section, comp_flag
        return None, None, None
    except Exception as e:
        logging.error(f"Error reading {promo_path}: {e}")
        return None, None, None

def extract_pi_section_text(pi_path, section_type):
    """
    From a PI JSON file, extract the text from the section whose "section_type" matches section_type (case-insensitive).
    Returns an empty string if not found.
    """
    try:
        with open(pi_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "sections" in data:
            for sec in data["sections"]:
                if sec.get("section_type", "").strip().upper() == section_type.upper():
                    t = sec.get("text", "").strip()
                    if t and t.upper() != "N/A":
                        return t
        logging.warning(f"Section '{section_type}' not found in {pi_path}.")
        return ""
    except Exception as e:
        logging.error(f"Error reading {pi_path}: {e}")
        return ""

def pair_positive(promos, pis):
    """
    Create positive pairs by matching a promo file with its corresponding PI file.
    For each promo file, extract (promo_text, referenced_PI_section, completeness_flag)
    and then find the matching PI file by drug and identifier.
    """
    pairs = []
    pi_lookup = {}
    for pi_path in pis:
        drug, num = extract_id_info(os.path.basename(pi_path))
        if drug is None or num is None:
            continue
        pi_lookup[(drug, num)] = pi_path
    for promo_path in promos:
        promo_name = os.path.basename(promo_path)
        drug, num = extract_id_info(promo_name)
        if drug is None or num is None:
            continue
        promo_text, ref_section, comp_flag = extract_data_source_field(promo_path)
        if promo_text is None or ref_section is None:
            continue
        if (drug, num) in pi_lookup:
            pi_path = pi_lookup[(drug, num)]
            pi_text = extract_pi_section_text(pi_path, ref_section)
            if promo_text and pi_text:
                pairs.append({
                    "promo_text": promo_text,
                    "pi_text": pi_text,
                    "matching_label": 1,
                    "promo_file": promo_name,
                    "pi_file": os.path.basename(pi_path),
                    "ref_section": ref_section,
                    "completeness_flag": comp_flag
                })
        else:
            logging.warning(f"No matching PI file for {promo_path}.")
    return pairs

def pair_negative(promos, pis, negative_ratio=1.0):
    """
    Create negative pairs by pairing each promo (with a valid data_source) with a PI file from a different drug.
    """
    pairs = []
    pi_by_drug = {}
    for pi_path in pis:
        drug, _ = extract_id_info(os.path.basename(pi_path))
        if drug is None:
            continue
        pi_by_drug.setdefault(drug, []).append(pi_path)
    for promo_path in promos:
        promo_name = os.path.basename(promo_path)
        drug, _ = extract_id_info(promo_name)
        promo_text, ref_section, comp_flag = extract_data_source_field(promo_path)
        if promo_text is None or ref_section is None:
            continue
        candidate_pis = []
        for d, files in pi_by_drug.items():
            if d != drug:
                for f in files:
                    pi_text = extract_pi_section_text(f, ref_section)
                    if pi_text:
                        candidate_pis.append(f)
        n_neg = max(1, int(negative_ratio))
        if candidate_pis:
            selected = random.sample(candidate_pis, min(n_neg, len(candidate_pis)))
            for neg_pi in selected:
                pi_text = extract_pi_section_text(neg_pi, ref_section)
                if not pi_text:
                    continue
                pairs.append({
                    "promo_text": promo_text,
                    "pi_text": pi_text,
                    "matching_label": 0,
                    "promo_file": promo_name,
                    "pi_file": os.path.basename(neg_pi),
                    "ref_section": ref_section,
                    "completeness_flag": comp_flag
                })
    return pairs

def build_training_dataframe(promo_compliant_dir, promo_noncompliant_dir, pi_compliant_dir, pi_noncompliant_dir, negative_ratio):
    """
    Build a training DataFrame by pairing promos with corresponding PI files from compliant and noncompliant directories.
    """
    promos_compliant = glob.glob(os.path.join(promo_compliant_dir, "*.json"))
    promos_noncompliant = glob.glob(os.path.join(promo_noncompliant_dir, "*.json"))
    pis_compliant = glob.glob(os.path.join(pi_compliant_dir, "*.json"))
    pis_noncompliant = glob.glob(os.path.join(pi_noncompliant_dir, "*.json"))
    
    pos_pairs = pair_positive(promos_compliant, pis_compliant) + pair_positive(promos_noncompliant, pis_noncompliant)
    neg_pairs = pair_negative(promos_compliant, pis_compliant, negative_ratio) + pair_negative(promos_noncompliant, pis_noncompliant, negative_ratio)
    
    all_pairs = pos_pairs + neg_pairs
    df = pd.DataFrame(all_pairs)
    if df.empty:
        logging.error("No pairs were found. Check your directories, file naming, and that valid data_source fields exist.")
        sys.exit(1)
    logging.info(f"Built training DataFrame with {len(df)} pairs. Columns: {df.columns.tolist()}")
    logging.info(f"First few rows:\n{df.head()}")
    return df

####################################
# Data Augmentation Functions
####################################
def augment_text(text):
    try:
        from nlpaug.augmenter.word import ContextualWordEmbsAug
        augmenter = ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        aug = augmenter.augment(text)
        if isinstance(aug, list):
            aug = " ".join(aug)
        return aug
    except Exception as e:
        logging.error(f"Augmentation error: {e}")
        return text

def apply_data_augmentation(df, augment_flag):
    """If enabled, augment the promo_text and replicate associated fields."""
    if not augment_flag:
        return df
    augmented_rows = []
    for _, row in df.iterrows():
        augmented_rows.append(row)
        new_row = row.copy()
        new_row["promo_text"] = augment_text(row["promo_text"])
        augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)

####################################
# Cross-Encoder Dataset
####################################
class CrossEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Concatenate promo and PI texts with [SEP]
        inputs = self.tokenizer.encode_plus(
            row["promo_text"],
            row["pi_text"],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        label = int(row["matching_label"])
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

####################################
# Training Function Using Hugging Face Trainer
####################################
def train_cross_encoder(args):
    # Build training DataFrame.
    df = build_training_dataframe(args.promo_compliant_dir, args.promo_noncompliant_dir,
                                  args.pi_compliant_dir, args.pi_noncompliant_dir, args.negative_ratio)
    logging.info(f"Built training DataFrame with {len(df)} pairs.")
    
    # Apply data augmentation if enabled.
    df = df if not args.augment else apply_data_augmentation(df, True)
    
    # Split into training+validation and test sets.
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["matching_label"])
    logging.info(f"Train+Val samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Initialize tokenizer and model.
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Create datasets.
    train_dataset = CrossEncoderDataset(train_df, tokenizer, max_length=args.max_length)
    test_dataset = CrossEncoderDataset(test_df, tokenizer, max_length=args.max_length)
    
    # Setup training arguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        disable_tqdm=False
    )
    
    # Define compute_metrics.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model.
    trainer.train()
    
    # Evaluate and log results.
    eval_result = trainer.evaluate()
    logging.info(f"Test evaluation results: {eval_result}")
    
    # Save the model.
    trainer.save_model(args.model_output)
    logging.info(f"Final model saved to {args.model_output}")
    
    # Get predictions and save evaluation CSV.
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)
    test_df = test_df.copy()
    test_df["predicted_label"] = preds
    test_df["predicted_probability"] = np.max(predictions_output.predictions, axis=1)
    eval_csv_path = os.path.join(args.output_dir, args.output_details_csv)
    test_df.to_csv(eval_csv_path, index=False)
    logging.info(f"Evaluation details CSV saved to {eval_csv_path}")

####################################
# Main and Argument Parsing
####################################
def main():
    parser = argparse.ArgumentParser(
        description="PI Section Cross-Encoder Training: Fine-tune a cross-encoder to decide if a promo's section matches the corresponding PI section."
    )
    parser.add_argument("--promo_compliant_dir", type=str, required=True,
                        help="Directory containing compliant promo JSON files.")
    parser.add_argument("--promo_noncompliant_dir", type=str, required=True,
                        help="Directory containing noncompliant promo JSON files.")
    parser.add_argument("--pi_compliant_dir", type=str, required=True,
                        help="Directory containing compliant PI JSON files.")
    parser.add_argument("--pi_noncompliant_dir", type=str, required=True,
                        help="Directory containing noncompliant PI JSON files.")
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Ratio of negative pairs to positive pairs (default 1.0).")
    parser.add_argument("--model_output", type=str, default="pi_section_matching_model",
                        help="Directory path to save the final model.")
    parser.add_argument("--output_details_csv", type=str, default="pi_section_matching_evaluation_details.csv",
                        help="Filename for evaluation details CSV (saved in output_dir).")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output plots, logs, and CSV files.")
    parser.add_argument("--pca_components", type=int, default=20,
                        help="Number of PCA components (not directly used in cross-encoder training).")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use as hold-out test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenizer.")
    parser.add_argument("--augment", type=bool, default=True,
                        help="If True, perform data augmentation on promo text.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda.")
    args = parser.parse_args()
    train_cross_encoder(args)

if __name__ == "__main__":
    main()
