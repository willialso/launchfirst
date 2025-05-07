#!/usr/bin/env python3
"""
Train Risk–Benefit Classifier

Description:
  - Loads a CSV file with columns "text" and "label" (with labels 0 for benefit and 1 for risk).
  - Computes embeddings for each text using SciBERT.
  - Splits data into training and validation sets (with stratification).
  - Optionally applies SMOTE to balance classes.
  - Trains a simple feed-forward classifier (BenefitRiskClassifier) with early stopping.
  - Saves the best model weights.
  
Usage:
  python train_benefit_risk_model.py --input_csv risk_benefit_data.csv --model_output benefit_risk_model.pt --epochs 20 --batch_size 32 --lr 1e-3 --hidden_dim 64 --patience 3 [--smote]
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the classifier architecture
class BenefitRiskClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super(BenefitRiskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.net(x)

# Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def main():
    parser = argparse.ArgumentParser(description="Train Risk–Benefit Classifier")
    parser.add_argument("--input_csv", required=True, help="Path to CSV with columns: text,label")
    parser.add_argument("--model_output", default="benefit_risk_model.pt", help="Path to save the trained model weights.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the classifier.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs with no improvement before early stopping.")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE to balance classes on the training set.")
    args = parser.parse_args()

    logging.info(f"Loading CSV data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns or "label" not in df.columns:
        logging.error("CSV must contain 'text' and 'label' columns.")
        return

    texts = df["text"].astype(str).tolist()
    labels = df["label"].values  # expecting numeric labels: 0=benefit, 1=risk

    unique_classes = np.unique(labels)
    logging.info(f"Unique classes in dataset: {unique_classes}")
    if len(unique_classes) < 2:
        logging.error("Error: Only one class found. Cannot train a 2-class classifier.")
        return

    logging.info("Loading SciBERT model for embeddings...")
    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")

    logging.info("Computing embeddings for all texts...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings)

    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=args.test_size, stratify=labels, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")

    # Optionally apply SMOTE on the training set.
    if args.smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE, training set shape: {X_train.shape}")

    # Create DataLoaders
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = BenefitRiskClassifier(input_dim=embeddings.shape[1], hidden_dim=args.hidden_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    logging.info("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        logging.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model weights.
            torch.save(model.state_dict(), args.model_output)
            logging.info(f"New best model saved at epoch {epoch+1}")
        else:
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= args.patience:
                logging.info("Early stopping triggered.")
                break

    # Final evaluation on validation set
    model.load_state_dict(torch.load(args.model_output))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    logging.info("Validation Classification Report (0=benefit, 1=risk):")
    logging.info("\n" + classification_report(all_labels, all_preds, target_names=["benefit", "risk"]))

    logging.info("Training complete. Best model saved to " + args.model_output)

if __name__ == "__main__":
    main()
