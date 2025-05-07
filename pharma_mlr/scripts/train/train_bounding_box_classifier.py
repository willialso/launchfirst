#!/usr/bin/env python3
"""
Train Bounding Box Risk-Benefit Classifier

This script:
  1. Loads a CSV with manually labeled bounding boxes (risk vs. benefit).
  2. Combines text embeddings (via SciBERT) and geometric features (area, prominence, etc.) 
     into a single feature vector.
  3. Trains a Logistic Regression classifier (or MLP) to predict risk (1) or benefit (0).
  4. Evaluates on a test split, then retrains on the full dataset and saves the model.

Usage Example:
  python train_bounding_box_classifier.py \
    --input_csv /app/outputs/bbox_merged_with_labels.csv \
    --model_output /app/models/bbox_risk_benefit_classifier.pkl \
    --test_size 0.2
"""

import argparse
import csv
import json
import logging
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train Bounding Box Risk-Benefit Classifier")
    parser.add_argument("--input_csv", required=True, help="CSV file with bounding box rows, text, geometry, and manual_label")
    parser.add_argument("--model_output", required=True, help="Path to save the trained model (pkl file)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing (default 0.2)")
    parser.add_argument("--text_column", default="text", help="Name of the column containing bounding box text (default 'text')")
    parser.add_argument("--label_column", default="manual_label", help="Name of the column with the label (default 'manual_label')")
    args = parser.parse_args()

    logging.info(f"Loading bounding box CSV from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded {len(df)} rows.")

    # Filter out rows that don't have a valid label (risk or benefit)
    # If you used "risk" or "benefit" as strings, you can do:
    df = df[df[args.label_column].isin(["risk", "benefit"])]
    logging.info(f"After filtering, {len(df)} labeled bounding boxes remain.")

    # Map labels to numeric
    label_map = {"risk": 1, "benefit": 0}
    df["label"] = df[args.label_column].map(label_map)

    # We'll combine geometry (area, prominence) plus text embeddings
    # If you also computed width, height, etc., you can include them too.
    geometry_features = []
    possible_geo_cols = ["area", "prominence", "width", "height"]
    for col in possible_geo_cols:
        if col in df.columns:
            geometry_features.append(col)
    logging.info(f"Using geometry columns: {geometry_features}")

    # We'll embed the bounding box text with SciBERT
    logging.info("Loading SciBERT for text embedding...")
    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")

    # Convert text to embeddings
    texts = df[args.text_column].fillna("").tolist()
    logging.info(f"Embedding {len(texts)} bounding box texts...")
    text_embeddings = embedder.encode(texts)  # shape (n_samples, 768)

    # Convert geometry to numpy
    if geometry_features:
        geo_data = df[geometry_features].fillna(0).values  # shape (n_samples, n_geo)
        # We'll just concatenate geometry + text embeddings
        X = np.hstack([text_embeddings, geo_data])
        logging.info(f"Feature dimension = {text_embeddings.shape[1]} + {geo_data.shape[1]} = {text_embeddings.shape[1] + geo_data.shape[1]}")
    else:
        X = text_embeddings
        logging.info(f"Feature dimension = {text_embeddings.shape[1]} (no geometry columns found).")

    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logging.info(f"Training set shape: {X_train.shape}, test set shape: {X_test.shape}")

    # Train a Logistic Regression
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["benefit", "risk"])
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n" + report)

    # Retrain on full data
    clf.fit(X, y)

    # Save the model
    # We might store it in a dict if we want to keep track of geometry_features
    model_obj = {
        "clf": clf,
        "geometry_features": geometry_features
    }
    joblib.dump(model_obj, args.model_output)
    logging.info(f"Trained bounding box classifier saved to {args.model_output}")

if __name__ == "__main__":
    main()
