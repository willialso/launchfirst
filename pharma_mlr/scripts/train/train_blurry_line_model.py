#!/usr/bin/env python3
"""
Blurry Line Training Script

Description:
  - Loads data from a CSV file (e.g., blurry_line_data.csv) that contains:
       * 'combined_text' (the extracted text from the JSON),
       * 'final_label' (either "blurry" or "non_blurry"),
       * and optionally meta features such as 'text_length'.
  - Maps final_label: "blurry" -> 1 (blurry/non-compliant) and any other value (e.g., "non_blurry") -> 0 (compliant).
  - For each sample, splits the text into chunks (approximately chunk_size words) and averages their embeddings using SciBERT.
  - Uses a stratified train/test split for an initial evaluation.
  - Trains a PyTorch logistic regression model and saves the final model as a .pt file.

Usage Example:
  python train_blurry_line_model.py \
    --csv_file blurry_line_data.csv \
    --chunk_size 300 \
    --test_size 0.2
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    if 'combined_text' in df.columns:
        texts = df['combined_text'].tolist()
    elif 'eff_claim_text' in df.columns:
        texts = df['eff_claim_text'].tolist()
    else:
        raise ValueError("CSV must contain either 'combined_text' or 'eff_claim_text' columns.")
    
    # Map final_label: "blurry" -> 1, else -> 0
    labels = df['final_label'].apply(lambda x: 1 if x.strip().lower() == "blurry" else 0).values
    return texts, labels

def chunk_text(text, chunk_size=300):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_embedding(text, embedder, chunk_size=300):
    """
    Splits text into chunks, computes embeddings for each chunk,
    and returns the average embedding.
    """
    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return None
    embeddings = embedder.encode(chunks)
    return np.mean(embeddings, axis=0)

# ------------------------------
# PyTorch Dataset
# ------------------------------
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.embeddings[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ------------------------------
# PyTorch Logistic Regression Model
# ------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# ------------------------------
# Training and Evaluation Functions
# ------------------------------
def train_model(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["features"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# ------------------------------
# Main Routine
# ------------------------------
def main(args):
    logging.info(f"Loading data from: {args.csv_file}")
    texts, labels = load_csv_data(args.csv_file)
    logging.info(f"Number of examples: {len(texts)}")
    logging.info(f"Class distribution: {dict(pd.Series(labels).value_counts())}")
    
    # Initialize SciBERT embedder.
    logging.info("Initializing SciBERT embedder...")
    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")
    
    # Generate embeddings for each sample.
    embeddings = []
    for idx, text in enumerate(texts):
        emb = generate_embedding(text, embedder, chunk_size=args.chunk_size)
        if emb is None:
            logging.warning(f"Empty embedding for sample {idx}; substituting zeros.")
            emb = np.zeros(768)  # Assume SciBERT embedding size is 768.
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    logging.info(f"Embeddings shape: {embeddings.shape}")
    
    # Split data for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=args.test_size, random_state=42, stratify=labels)
    logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create datasets and dataloaders.
    train_dataset = EmbeddingDataset(X_train, y_train)
    test_dataset = EmbeddingDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(args.device)
    
    # Define model.
    input_dim = embeddings.shape[1]
    model = LogisticRegressionModel(input_dim=input_dim, num_classes=2)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train model.
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_model(model, optimizer, criterion, train_loader, device)
        logging.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    
    # Evaluate on test set.
    test_loss, test_preds, test_labels = evaluate_model(model, criterion, test_loader, device)
    test_acc = accuracy_score(test_labels, test_preds)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["non_blurry", "blurry"]))
    
    # Retrain on full data.
    full_dataset = EmbeddingDataset(embeddings, labels)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
    best_full_loss = float('inf')
    for epoch in range(args.epochs):
        loss = train_model(model, optimizer, criterion, full_loader, device)
        logging.info(f"Full Training Epoch {epoch+1}: Loss {loss:.4f}")
        if loss < best_full_loss:
            best_full_loss = loss
            best_full_state = model.state_dict()
    model.load_state_dict(best_full_state)
    
    # Save final model as .pt file.
    torch.save(model.state_dict(), args.model_output)
    logging.info(f"Final model saved to {args.model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Blurry Line Classifier from CSV data")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV file (e.g., blurry_line_data.csv)")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Approximate number of words per chunk for embedding generation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for evaluation split")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Path to save the final model (e.g., advanced_blurry_line_model.pt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda")
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
