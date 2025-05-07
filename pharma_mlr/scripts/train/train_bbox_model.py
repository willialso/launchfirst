#!/usr/bin/env python3
"""
Training script for a bounding box segmentation model.

This model takes as input a feature vector constructed by concatenating:
  - A 768-dimensional text embedding (from SciBERT) of the chunk text.
  - 4 numeric features (aggregated bounding box coordinates for the chunk).
Total input dimension = 772.

It outputs:
  - 4 regression values (refined bounding box coordinates: xmin, ymin, xmax, ymax).
  - 5 classification logits corresponding to the segment classes:
      PROMO_HEADLINE, EFFICACY_CLAIMS, CLINICAL_STUDIES, REFERENCE, IMPORTANT_SAFETY_INFORMATION.

The training data is assumed to be in a CSV file with columns:
  document_id, chunk_text, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, manual_label

Usage:
  python scripts/train/train_bbox_model.py --csv_path /app/outputs/training_chunks.csv --output_model bbox_model.pt --epochs 50 --batch_size 16 --lr 1e-4
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label mapping for our 5 classes.
# The order is important: the model's final classification layer will output 5 logits.
LABELS = ["PROMO_HEADLINE", "EFFICACY_CLAIMS", "CLINICAL_STUDIES", "REFERENCE", "IMPORTANT_SAFETY_INFORMATION"]

# Create a PyTorch Dataset.
class PromoDataset(Dataset):
    def __init__(self, df, embedder, scaler):
        """
        df: DataFrame with columns: chunk_text, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, manual_label
        embedder: SentenceTransformer instance (SciBERT)
        scaler: A StandardScaler fitted on the bounding box features.
        """
        self.df = df
        self.embedder = embedder
        self.scaler = scaler

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['manual_label'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Compute text embedding (fixed 768-dim vector)
        text = row['chunk_text']
        emb = self.embedder.encode(text, convert_to_tensor=True)
        # Extract bounding box features (4 numbers)
        bbox = np.array([row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']], dtype=np.float32)
        bbox = self.scaler.transform(bbox.reshape(1, -1))[0]  # scale features
        # Concatenate text embedding with bbox features (total 768+4 = 772)
        features = torch.cat([emb, torch.tensor(bbox, dtype=torch.float32)])
        # Targets:
        # For regression: use the original bbox as target (we can also choose to use refined bbox if available)
        target_bbox = torch.tensor([row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']], dtype=torch.float32)
        # For classification: target label
        target_label = torch.tensor(row['label_encoded'], dtype=torch.long)
        return features, target_bbox, target_label

# Define the model with two heads: one for bounding box regression, one for classification.
class BBoxModel(nn.Module):
    def __init__(self, input_dim=772, hidden_dim=256, num_classes=5):
        super(BBoxModel, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Regression head: 4 outputs (bbox coordinates)
        self.regressor = nn.Linear(hidden_dim, 4)
        # Classification head: num_classes outputs (logits)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        shared = self.shared_fc(x)
        bbox_pred = self.regressor(shared)
        class_logits = self.classifier(shared)
        return bbox_pred, class_logits

def train_model(csv_path, output_model, epochs=50, batch_size=16, lr=1e-4):
    # Load training CSV
    df = pd.read_csv(csv_path)
    # Ensure necessary columns exist.
    required_cols = ["chunk_text", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "manual_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Fit a scaler on bounding box features.
    bbox_features = df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(bbox_features)
    
    # Load SciBERT model for embeddings.
    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")
    
    dataset = PromoDataset(df, embedder, scaler)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = BBoxModel(input_dim=772, hidden_dim=256, num_classes=len(LABELS)).to(device)
    # Loss functions: MSE for bbox regression and CrossEntropy for classification.
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, target_bbox, target_label in dataloader:
            features = features.to(device)
            target_bbox = target_bbox.to(device)
            target_label = target_label.to(device)
            
            optimizer.zero_grad()
            bbox_pred, class_logits = model(features)
            loss_reg = criterion_reg(bbox_pred, target_bbox)
            loss_cls = criterion_cls(class_logits, target_label)
            loss = loss_reg + loss_cls
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), output_model)
    print(f"Model saved to {output_model}")

def main():
    parser = argparse.ArgumentParser(description="Train bounding box segmentation model.")
    parser.add_argument("--csv_path", required=True,
                        help="Path to the training CSV file (e.g., /app/outputs/training_chunks.csv)")
    parser.add_argument("--output_model", required=True,
                        help="Path to save the trained model (e.g., bbox_model.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_model(args.csv_path, args.output_model, args.epochs, args.batch_size, args.lr)

if __name__ == "__main__":
    main()
