#!/usr/bin/env python3
"""
Advanced FDA Title 21 Overlay Training

This script trains an overlay model to predict the FDA Title 21 classification
(e.g., "violation", "no_violation", or specific violation codes) for chunk‑level claims.
It expects a CSV with the following columns:
  - combined_text: The full text (e.g., claim text combined with any appended violation_reason).
  - final_label: The FDA Title 21 label.
  - text_length: A numeric feature (e.g., length of the text).
  - matched_rule: A string indicating the matched FDA rule (used for evaluation).

The script computes features from combined_text using:
  • Averaged embeddings from SciBERT and RoBERTa.
  • TF‑IDF features (reduced via PCA).
  • VADER sentiment scores.
  • Normalized text_length.
These features are concatenated into one final vector.

An MLP classifier is trained using K‑fold cross‑validation with early stopping,
and Monte Carlo dropout is applied for uncertainty estimation. A learning curve (from fold 1),
an evaluation CSV (with sample index, true label, predicted label, predicted probability, uncertainty,
and matched_rule), and the final model (saved as a .pt file) are produced.

Usage Example (from docker):
  docker run --rm -v ~/Desktop/pharma_mlr:/app -w /app/scripts/train my-pharma-mlr \
    python advanced_title21_overlay_training.py \
      --csv_file /app/outputs/title21_claims.csv \
      --model_output /app/models/advanced_title21_overlay_model.pt \
      --output_csv advanced_title21_overlay_evaluation.csv \
      --output_dir /app/outputs \
      --test_size 0.2 \
      --epochs 20 \
      --lr 1e-3 \
      --hidden_dim 128 \
      --batch_size 16 \
      --pca_components 10 \
      --kfold 5 \
      --patience 3 \
      --mc_iterations 10 \
      --seed 42 \
      --device cpu
"""

import argparse, logging, os, random, sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

####################################
# Data Loading and Preprocessing
####################################
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    required = {"combined_text", "final_label", "text_length", "matched_rule"}
    if not required.issubset(set(df.columns)):
        logging.error(f"CSV must contain columns: {required}")
        sys.exit(1)
    logging.info(f"Loaded {len(df)} samples from CSV.")
    return df

def normalize_numeric_feature(series):
    min_val, max_val = series.min(), series.max()
    if max_val - min_val != 0:
        return (series - min_val) / (max_val - min_val)
    return series * 0.0

def map_labels(df):
    unique_labels = sorted(df["final_label"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df["label"] = df["final_label"].map(label_map)
    logging.info(f"Label mapping: {label_map}")
    return df, label_map

####################################
# Feature Computation Functions
####################################
def compute_averaged_embeddings(texts):
    logging.info("Computing SciBERT embeddings...")
    model_sci = SentenceTransformer("allenai/scibert_scivocab_uncased")
    emb_sci = model_sci.encode(texts, show_progress_bar=True)
    
    logging.info("Computing RoBERTa embeddings...")
    model_robo = SentenceTransformer("roberta-base")
    emb_robo = model_robo.encode(texts, show_progress_bar=True)
    
    avg_emb = (emb_sci + emb_robo) / 2.0
    return avg_emb

def compute_tfidf_features(texts, n_components=10):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(tfidf.toarray())
    return reduced

def compute_sentiment_scores(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)["compound"] for text in texts]
    return np.array(scores).reshape(-1, 1)

def build_feature_matrix(df, pca_components):
    texts = df["combined_text"].tolist()
    avg_emb = compute_averaged_embeddings(texts)
    tfidf_feat = compute_tfidf_features(texts, n_components=pca_components)
    sent_scores = compute_sentiment_scores(texts)
    norm_length = normalize_numeric_feature(df["text_length"]).values.reshape(-1, 1)
    X = np.concatenate([avg_emb, tfidf_feat, sent_scores, norm_length], axis=1)
    return X

####################################
# Neural Network Model for Multi-Class Classification
####################################
class FDATitle21Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_p=0.3):
        super(FDATitle21Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        return self.out(x)

def monte_carlo_dropout(model, x, mc_iterations=10):
    model.train()  # keep dropout active during inference
    preds = []
    with torch.no_grad():
        for _ in range(mc_iterations):
            out = model(x)
            preds.append(out.cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    var_pred = preds.var(axis=0)
    return mean_pred, var_pred

####################################
# Custom Collate Function
####################################
def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        # If key is "matched_rule", leave it as a list.
        if key == "matched_rule":
            collated[key] = [d[key] for d in batch]
        else:
            # If the first element is already a tensor, stack it.
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([d[key] for d in batch])
            # If it's a number (int, float, or NumPy scalar), convert to float.
            elif isinstance(batch[0][key], (int, float)) or np.isscalar(batch[0][key]):
                collated[key] = torch.tensor([float(d[key]) for d in batch], dtype=torch.float)
            else:
                # Otherwise, leave as list.
                collated[key] = [d[key] for d in batch]
    return collated

####################################
# Custom Dataset
####################################
class ClaimsDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, source_files):
        self.features = features
        self.labels = labels
        self.source_files = source_files
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "matched_rule": self.source_files[idx]
        }

####################################
# Training and Evaluation Functions
####################################
def train_model(model, optimizer, criterion, data_loader, device):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        inputs = batch["features"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(data_loader.dataset)

def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)

####################################
# Main Training Routine with K-Fold CV
####################################
def main():
    parser = argparse.ArgumentParser(description="Advanced FDA Title 21 Overlay Classifier Training")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file with chunk-level claims and FDA Title 21 labels (columns: claim_text, fda_rule, combined_text, text_length, matched_rule)")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Path to save the final model (e.g., .pt file)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Filename for evaluation details CSV (saved in output_dir)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output plots and CSV files")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use as hold-out test set")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for the MLP")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--pca_components", type=int, default=10,
                        help="Number of PCA components for TF-IDF features")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--mc_iterations", type=int, default=10,
                        help="Number of Monte Carlo dropout iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Load CSV data.
    df = load_csv_data(args.csv_file)
    if "combined_text" not in df.columns or "final_label" not in df.columns:
        logging.error("CSV must contain 'combined_text' and 'final_label' columns.")
        sys.exit(1)
    logging.info(f"Using {len(df)} samples.")
    
    # Map final_label to numeric classes.
    df, label_map = map_labels(df)
    num_classes = len(label_map)
    
    # Build feature matrix.
    X = build_feature_matrix(df, args.pca_components)
    logging.info(f"Final feature matrix shape: {X.shape}")
    y = df["label"].values
    source_files = df["matched_rule"].tolist()
    logging.info(f"Target distribution: {Counter(y)}")
    
    # Split into train+validation and hold-out test sets.
    X_train_val, X_test, y_train_val, y_test, sf_train_val, sf_test = train_test_split(
        X, y, source_files, test_size=args.test_size, random_state=args.seed, stratify=y)
    logging.info(f"Training+Validation: {len(y_train_val)} samples; Hold-out test: {len(y_test)} samples")
    
    device = torch.device(args.device)
    
    # K-Fold cross-validation on train+validation set.
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_errors = []
    train_losses_all = []
    val_losses_all = []
    eval_details = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val), 1):
        logging.info(f"Starting fold {fold}/{args.kfold}")
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        sf_val = [sf_train_val[i] for i in val_idx]
        
        train_dataset = ClaimsDataset(X_train, y_train, [sf_train_val[i] for i in train_idx])
        val_dataset = ClaimsDataset(X_val, y_val, sf_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
        
        model = FDATitle21Classifier(input_dim=X.shape[1], hidden_dim=args.hidden_dim, num_classes=num_classes, dropout_p=0.3)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        fold_train_losses = []
        fold_val_losses = []
        for epoch in range(args.epochs):
            train_loss = train_model(model, optimizer, criterion, train_loader, device)
            val_loss, val_preds, val_labels = evaluate_model(model, criterion, val_loader, device)
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            logging.info(f"Fold {fold} Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1} for fold {fold}")
                break
        model.load_state_dict(best_state)
        _, fold_preds, fold_labels = evaluate_model(model, criterion, val_loader, device)
        fold_pred_classes = np.argmax(fold_preds, axis=1)
        fold_error = np.mean((fold_pred_classes - fold_labels)**2)
        fold_errors.append(fold_error)
        logging.info(f"Fold {fold} Error: {fold_error:.4f}")
        
        # Monte Carlo dropout for uncertainty estimation on validation set.
        mean_logits, var_preds = monte_carlo_dropout(model, torch.tensor(X_val, dtype=torch.float).to(device), mc_iterations=args.mc_iterations)
        for i, idx in enumerate(val_idx):
            pred_prob = torch.softmax(torch.tensor(mean_logits[i]), dim=0).cpu().numpy()
            eval_details.append({
                "sample_index": idx,
                "matched_rule": sf_train_val[idx],
                "true_label": int(y_train_val[idx]),
                "predicted_label": int(np.argmax(pred_prob)),
                "predicted_probability": float(np.max(pred_prob)),
                "uncertainty": float(var_preds[i].max())
            })
        train_losses_all.append(fold_train_losses)
        val_losses_all.append(fold_val_losses)
    
    avg_cv_error = np.mean(fold_errors)
    logging.info(f"Average Cross-Validation Error: {avg_cv_error:.4f}")
    
    # Plot learning curve for fold 1.
    epochs_range = range(1, len(train_losses_all[0]) + 1)
    plt.figure(figsize=(8,6))
    plt.plot(epochs_range, train_losses_all[0], label="Train Loss")
    plt.plot(epochs_range, val_losses_all[0], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Fold 1)")
    plt.legend()
    lc_path = os.path.join(args.output_dir, "advanced_fda_title21_learning_curve.png")
    plt.savefig(lc_path)
    plt.close()
    logging.info(f"Learning curve saved to {lc_path}")
    
    # Retrain final model on full training+validation set.
    full_dataset = ClaimsDataset(X_train_val, y_train_val, sf_train_val)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    best_full_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        loss = train_model(model, optimizer, criterion, full_loader, device)
        logging.info(f"Full Training Epoch {epoch+1}: Loss {loss:.4f}")
        if loss < best_full_loss:
            best_full_loss = loss
            best_full_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            logging.info(f"Early stopping in full training at epoch {epoch+1}")
            break
    model.load_state_dict(best_full_state)
    torch.save(model.state_dict(), args.model_output)
    logging.info(f"Final model saved to {args.model_output}")
    
    # Evaluate on hold-out test set.
    test_dataset = ClaimsDataset(X_test, y_test, sf_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    test_loss, test_preds, test_labels = evaluate_model(model, criterion, test_loader, device)
    test_pred_classes = np.argmax(test_preds, axis=1)
    test_acc = accuracy_score(test_labels, test_pred_classes)
    report = classification_report(test_labels, test_pred_classes, target_names=list(label_map.keys()))
    logging.info(f"Hold-Out Test Loss: {test_loss:.4f}")
    logging.info(f"Hold-Out Test Accuracy: {test_acc:.4f}")
    logging.info("Classification Report:\n" + report)
    
    # Save evaluation details CSV.
    eval_df = pd.DataFrame(eval_details)
    eval_csv_path = os.path.join(args.output_dir, args.output_csv)
    eval_df.to_csv(eval_csv_path, index=False)
    logging.info(f"Evaluation details CSV saved to {eval_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced FDA Title 21 Overlay Classifier Training")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file with chunk-level claims and FDA Title 21 labels (columns: claim_text, fda_rule, combined_text, text_length, matched_rule)")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Path to save the final model (e.g., .pt file)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Filename for evaluation details CSV (saved in output_dir)")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save output plots and CSV files")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use as hold-out test set")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for the MLP")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--pca_components", type=int, default=10,
                        help="Number of PCA components for TF-IDF features")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--mc_iterations", type=int, default=10,
                        help="Number of Monte Carlo dropout iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda")
    args = parser.parse_args()
    main()
