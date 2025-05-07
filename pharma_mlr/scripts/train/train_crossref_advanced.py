#!/usr/bin/env python3
"""
Train Cross-Reference Classifier Using Advanced Embedding Features

This script:
  1. Loads a combined cross-reference training CSV (e.g., combined_crossref_training.csv) which is expected to contain:
       - promo_text: the text from the promo section.
       - best_pi_text: the best-matching text from the PI.
       - similarity: a CSV-generated similarity score.
       - manual_label: a binary label (0 = not supported; 1 = supported).
  2. Computes cosine similarity scores between promo_text and best_pi_text using two SentenceTransformer models:
       - SciBERT (model: "allenai/scibert_scivocab_uncased")
       - RoBERTa (model: "roberta-base")
  3. For each example, forms a feature vector:
         [csv_similarity, sci_cos_sim, roberta_cos_sim]
  4. Splits the data into training and test sets, trains a Logistic Regression classifier on these features, and prints evaluation metrics.
  5. Retrains on the full dataset and saves the final model to a pickle file.

Usage Example:
  python train_crossref_advanced.py \
    --input_csv /app/outputs/combined_crossref_training.csv \
    --model_output /app/models/crossref_model_advanced.pkl \
    --test_size 0.2
"""

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows from {csv_path}")
    # Keep only rows with a valid manual label (assumed to be 0 or 1)
    df = df[df["manual_label"].notna() & (df["manual_label"] != "")]
    logging.info(f"Filtered to {len(df)} examples with manual labels.")
    df["label"] = df["manual_label"].astype(int)
    return df

def compute_cosine_sim(text1, text2, embedder):
    emb1 = embedder.encode([text1])[0]
    emb2 = embedder.encode([text2])[0]
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return sim

def main():
    parser = argparse.ArgumentParser(description="Train Cross-Reference Classifier with Advanced Embedding Features")
    parser.add_argument("--input_csv", required=True, help="Path to the combined cross-reference training CSV")
    parser.add_argument("--model_output", required=True, help="Path to save the trained model (pkl file)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing (default 0.2)")
    args = parser.parse_args()

    # Load data
    df = load_data(args.input_csv)
    
    # Extract the required columns
    # We assume CSV has: promo_text, best_pi_text, similarity (CSV similarity), and label
    if not {"promo_text", "best_pi_text", "similarity", "label"}.issubset(set(df.columns)):
        logging.error("CSV is missing one or more required columns: promo_text, best_pi_text, similarity, manual_label")
        return

    # Use the CSV similarity score as the first feature.
    csv_sim = df["similarity"].astype(float).values.reshape(-1, 1)

    # Load two embedding models.
    logging.info("Loading SciBERT model...")
    scibert = SentenceTransformer("allenai/scibert_scivocab_uncased")
    logging.info("Loading RoBERTa model...")
    roberta = SentenceTransformer("roberta-base")

    # Compute cosine similarities using both models.
    sci_sims = []
    roberta_sims = []
    promo_texts = df["promo_text"].fillna("").tolist()
    pi_texts = df["best_pi_text"].fillna("").tolist()
    logging.info(f"Computing embeddings and cosine similarities for {len(promo_texts)} examples...")
    for promo, pi in zip(promo_texts, pi_texts):
        sci_sim = compute_cosine_sim(promo, pi, scibert)
        robo_sim = compute_cosine_sim(promo, pi, roberta)
        sci_sims.append(sci_sim)
        roberta_sims.append(robo_sim)
    sci_sims = np.array(sci_sims).reshape(-1, 1)
    roberta_sims = np.array(roberta_sims).reshape(-1, 1)

    # Combine features: [CSV similarity, SciBERT similarity, RoBERTa similarity]
    X = np.hstack([csv_sim, sci_sims, roberta_sims])
    y = df["label"].values
    logging.info(f"Feature vector shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=42, stratify=y)
    logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train Logistic Regression classifier
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["non-match", "match"])
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n" + report)

    # Retrain on full dataset
    clf.fit(X, y)

    # Save the model (store additional info if needed)
    model_obj = {
        "clf": clf,
        "feature_description": "Features = [csv_similarity, sciBERT_cosine, roberta_cosine]"
    }
    joblib.dump(model_obj, args.model_output)
    logging.info(f"Trained cross-reference model saved to {args.model_output}")

if __name__ == "__main__":
    main()
