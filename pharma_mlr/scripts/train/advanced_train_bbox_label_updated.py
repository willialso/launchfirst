#!/usr/bin/env python3
"""
Train a bounding box classifier using advanced text embeddings and numeric features.

This script:
  1. Loads a labeled CSV containing:
       - text, x_min, y_min, x_max, y_max, area, prominence, manual_label
  2. Computes a 768-dimensional embedding for the text using SciBERT (SentenceTransformer "allenai/scibert_scivocab_uncased").
  3. Concatenates the text embedding with the numeric features (total dimension = embedding_dim + 6).
  4. Trains a feed-forward neural network classifier to predict the manual label.
  5. Saves the trained model to a file (e.g., model.pt).

Usage Example:
  python advanced_train_bbox_label_updated.py --csv_path outputs/ocr_bbox_annotation.csv --output_model model.pt --epochs 50 --batch_size 16 --lr 1e-4
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCRDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, embedder):
        """
        texts: list of strings (OCR text)
        numeric_features: numpy array of shape (n_samples, n_numeric_features)
        labels: numpy array of integer labels
        embedder: SentenceTransformer model to encode texts
        """
        self.texts = texts
        self.numeric_features = numeric_features
        self.labels = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Compute embedding for the text at this index
        embedding = self.embedder.encode(self.texts[idx], convert_to_tensor=True)
        numeric = torch.tensor(self.numeric_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, numeric, label


class CombinedClassifier(nn.Module):
    def __init__(self, embedding_dim, numeric_dim, num_classes):
        super(CombinedClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, emb, numeric):
        # Concatenate text embedding and numeric features
        x = torch.cat([emb, numeric], dim=1)
        return self.fc(x)


def train_model(csv_path, output_model_path, epochs=50, batch_size=16, learning_rate=1e-4):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["text", "x_min", "y_min", "x_max", "y_max", "area", "prominence", "manual_label"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain the columns: {required_cols}")

    # Extract text and numeric features
    texts = df["text"].astype(str).tolist()
    numeric_features = df[["x_min", "y_min", "x_max", "y_max", "area", "prominence"]].values

    # Use manual_label as ground truth
    labels = df["manual_label"].astype(str).tolist()

    # Encode the labels as integers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Scale the numeric features
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)

    # Load SentenceTransformer model (SciBERT)
    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased", device=device)

    # Split data into train and validation sets
    texts_train, texts_val, num_train, num_val, y_train, y_val = train_test_split(
        texts, numeric_features_scaled, labels_encoded, test_size=0.2, random_state=42
    )

    # Create Dataset and DataLoader objects
    train_dataset = OCRDataset(texts_train, num_train, y_train, embedder)
    val_dataset = OCRDataset(texts_val, num_val, y_val, embedder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    embedding_dim = 768  # SciBERT outputs 768-dimensional embeddings
    numeric_dim = numeric_features_scaled.shape[1]
    num_classes = len(le.classes_)

    model = CombinedClassifier(embedding_dim, numeric_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for emb, numeric, labels_batch in train_loader:
            emb = emb.to(device)
            numeric = numeric.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(emb, numeric)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * emb.size(0)

        train_loss /= len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for emb, numeric, labels_batch in val_loader:
                emb = emb.to(device)
                numeric = numeric.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(emb, numeric)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * emb.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (preds == labels_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

    # Optionally, save the label encoder and scaler for inference
    # For example, using pickle (not shown here)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bounding box classifier using advanced embeddings")
    parser.add_argument("--csv_path", required=True, help="Path to the labeled CSV file")
    parser.add_argument("--output_model", required=True, help="Path to save the trained model (e.g., model.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train_model(args.csv_path, args.output_model, args.epochs, args.batch_size, args.lr)
