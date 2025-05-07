#!/usr/bin/env python3
"""
Advanced Iterative Training Script for Chunk-Level Classification with Overfitting Evaluation,
Early Stopping, Increased Regularization, Reduced Model Complexity, and Contextual Augmentation

This script:
  - Loads a CSV (with columns "text" and "label") from the outputs folder.
  - Applies data augmentation using contextual word embedding-based substitution.
  - Computes features using dual SentenceTransformer models (SciBERT and RoBERTa),
    TF-IDF (with PCA), VADER sentiment scores, and clustering features.
  - Combines these into a single feature matrix.
  - Splits the data into training+validation (80%) and hold-out test (20%).
  - Uses StratifiedKFold cross-validation with early stopping to train a feed-forward NN.
  - Retrains on full training+validation and evaluates on a hold-out test set.
  - Saves a learning curve plot and writes per-example evaluation details to CSV.
  
Usage Example (from docker):
  docker run --rm -v ~/Desktop/pharma_mlr:/app -w /app/scripts/train my-pharma-mlr \
    python advanced_iterative_train_chunk_level_overfit.py \
      --input_csv /app/outputs/training_examples.csv \
      --model_output /app/models/advanced_chunk_model.pt \
      --output_details_csv /app/outputs/advanced_evaluation_details.csv \
      --kfold 5 \
      --epochs 20 \
      --lr 0.001 \
      --hidden_dim 64 \
      --dropout 0.7 \
      --weight_decay 1e-4 \
      --patience 3 \
      --num_augment 1 \
      --chunk_size 300 \
      --mc_iterations 10 \
      --device cpu
"""

import argparse, csv, logging, os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##############################
# Data Processing & Feature Extraction
##############################
def load_csv_data(csv_path):
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.array(labels)

def augment_text(text, num_aug=1):
    """
    Uses contextual word embeddings for word substitution.
    Returns a list containing the original text and its augmented versions.
    Ensures that the result is a string.
    """
    from nlpaug.augmenter.word import ContextualWordEmbsAug
    augmenter = ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    augmented = [text]
    for _ in range(num_aug):
        try:
            aug_result = augmenter.augment(text)
            # If the result is a list, join it into a string.
            if isinstance(aug_result, list):
                aug_result = " ".join(aug_result)
            augmented.append(str(aug_result))
        except Exception as e:
            logging.error(f"Contextual augmentation error: {e}")
    return augmented

def compute_tfidf_features(texts, n_components=50):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(tfidf.toarray())

def compute_sentiment_scores(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = np.array([analyzer.polarity_scores(t)["compound"] for t in texts])
    return scores.reshape(-1, 1)

def compute_embeddings(texts, model_name):
    """
    Computes sentence embeddings.
    Ensures each element is a string and replaces empty texts with a single space.
    If using SciBERT ("allenai/scibert_scivocab_uncased"), builds a model with Transformer and Pooling.
    Otherwise, loads the model by name.
    """
    from sentence_transformers import SentenceTransformer, models
    processed_texts = [str(text).strip() if str(text).strip() != "" else " " for text in texts]
    if model_name == "allenai/scibert_scivocab_uncased":
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(model_name)
    return model.encode(processed_texts)

def compute_cluster_features(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return np.eye(n_clusters)[clusters]

def build_feature_matrix(texts, sci_embeddings, robo_embeddings, tfidf_feats, sent_scores, cluster_feats):
    features = []
    for i in range(len(texts)):
        sci = sci_embeddings[i]
        robo = robo_embeddings[i]
        cos_sim = cosine_similarity([sci], [robo])[0][0]
        combined = np.concatenate([sci, robo, [cos_sim], tfidf_feats[i], [sent_scores[i][0]], cluster_feats[i]])
        features.append(combined)
    return np.array(features)

##############################
# Model Definition: Classification with MC Dropout
##############################
class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_p):
        super(AdvancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def monte_carlo_dropout(model, x, mc_iterations=10):
    model.train()  # Enable dropout during inference
    preds = []
    with torch.no_grad():
        for _ in range(mc_iterations):
            logits = model(x)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.var(axis=0)

##############################
# Training, Cross-Validation & Hold-Out Test Pipeline
##############################
def train_and_evaluate(args):
    # Load CSV data
    texts, labels = load_csv_data(args.input_csv)
    logging.info(f"Loaded {len(texts)} examples. Class distribution: {Counter(labels)}")
    
    # Data augmentation
    aug_texts, aug_labels = [], []
    for text, label in zip(texts, labels):
        for t in augment_text(text, num_aug=args.num_augment):
            aug_texts.append(t)
            aug_labels.append(label)
    logging.info(f"After augmentation: {len(aug_texts)} examples. Distribution: {Counter(aug_labels)}")
    
    # Compute features using dual SentenceTransformer models
    logging.info("Computing SciBERT embeddings...")
    sci_embeddings = compute_embeddings(aug_texts, "allenai/scibert_scivocab_uncased")
    logging.info("Computing RoBERTa embeddings...")
    robo_embeddings = compute_embeddings(aug_texts, "roberta-base")
    logging.info("Computing TF-IDF features...")
    tfidf_feats = compute_tfidf_features(aug_texts, n_components=50)
    logging.info("Computing sentiment scores...")
    sent_scores = compute_sentiment_scores(aug_texts)
    logging.info("Computing clustering features...")
    cluster_feats = compute_cluster_features(sci_embeddings, n_clusters=5)
    
    # Build feature matrix and prepare labels
    X = build_feature_matrix(aug_texts, sci_embeddings, robo_embeddings, tfidf_feats, sent_scores, cluster_feats)
    y = np.array(aug_labels)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    # Split into training+validation and hold-out test sets
    X_train_val, X_test, y_train_val, y_test, texts_train_val, texts_test = train_test_split(
        X, y, aug_texts, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(f"Training+Validation: {X_train_val.shape[0]} samples; Hold-out test: {X_test.shape[0]} samples")
    
    # Cross-validation using StratifiedKFold
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    fold = 1
    fold_reports = []
    train_losses_all = []
    val_losses_all = []
    advanced_details = []  # For per-example evaluation details
    
    input_dim = X.shape[1]
    device = torch.device(args.device)
    
    for train_idx, val_idx in skf.split(X_train_val, y_train_val):
        logging.info(f"Starting fold {fold}/{args.kfold}")
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        texts_val = [texts_train_val[i] for i in val_idx]
        
        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        
        model = AdvancedClassifier(input_dim, args.hidden_dim, 2, args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
            logging.info(f"Fold {fold} Epoch {epoch+1}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}")
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_fold_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= args.patience:
                logging.info(f"Early stopping at epoch {epoch+1} for fold {fold}")
                break
        
        model.load_state_dict(best_fold_state)
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).argmax(dim=1)
        acc = accuracy_score(y_val, val_preds.cpu().numpy())
        fold_report = classification_report(y_val, val_preds.cpu().numpy(), output_dict=True)
        logging.info(f"Fold {fold} Accuracy: {acc:.4f}")
        fold_reports.append(fold_report)
        
        # Use MC dropout for uncertainty estimates and record details
        mean_preds, var_preds = monte_carlo_dropout(model, X_val_tensor, mc_iterations=args.mc_iterations)
        final_preds = mean_preds.argmax(axis=1)
        for i, idx in enumerate(val_idx):
            advanced_details.append({
                "text": texts_train_val[idx],
                "true_label": int(y_train_val[idx]),
                "predicted_label": int(final_preds[i]),
                "uncertainty": float(var_preds[i].max())
            })
        fold += 1

    # Save learning curve for fold 1
    outputs_dir = os.path.dirname(args.input_csv)
    learning_curve_path = os.path.join(outputs_dir, "learning_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses_all[0])+1), train_losses_all[0], label="Train Loss")
    plt.plot(range(1, len(val_losses_all[0])+1), val_losses_all[0], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Fold 1)")
    plt.legend()
    plt.savefig(learning_curve_path)
    plt.close()
    logging.info(f"Learning curve saved to {learning_curve_path}")
    
    avg_accuracy = np.mean([r["accuracy"] for r in fold_reports])
    logging.info(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
    
    # Retrain on full training+validation set
    logging.info("Retraining on full training+validation set...")
    X_train_val_tensor = torch.tensor(X_train_val, dtype=torch.float).to(device)
    y_train_val_tensor = torch.tensor(y_train_val, dtype=torch.long).to(device)
    final_model = AdvancedClassifier(input_dim, args.hidden_dim, 2, args.dropout).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        final_model.train()
        optimizer.zero_grad()
        outputs = final_model(X_train_val_tensor)
        loss = criterion(outputs, y_train_val_tensor)
        loss.backward()
        optimizer.step()
        logging.info(f"Full Training Epoch {epoch+1}: Loss {loss.item():.4f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = final_model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            logging.info(f"Early stopping in full training at epoch {epoch+1}")
            break
    final_model.load_state_dict(best_state)
    torch.save(final_model.state_dict(), args.model_output)
    logging.info(f"Final model saved to {args.model_output}")
    
    # Evaluate on hold-out test set
    final_model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        test_outputs = final_model(X_test_tensor)
        test_preds = test_outputs.argmax(dim=1)
    test_accuracy = accuracy_score(y_test, test_preds.cpu().numpy())
    logging.info(f"Hold-Out Test Accuracy: {test_accuracy:.4f}")
    
    for i in range(len(X_test)):
        advanced_details.append({
            "text": texts_test[i],
            "true_label": int(y_test[i]),
            "predicted_label": int(test_preds.cpu().numpy()[i]),
            "uncertainty": 0.0
        })
    
    with open(args.output_details_csv, "w", newline="", encoding="utf-8") as fout:
        fieldnames = ["text", "true_label", "predicted_label", "uncertainty"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in advanced_details:
            writer.writerow(row)
    
    return fold_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Iterative Training for Chunk-Level Classification with Overfitting Evaluation, "
                    "Early Stopping, Increased Regularization, Reduced Complexity, and Contextual Augmentation"
    )
    parser.add_argument("--input_csv", required=True, help="Path to training CSV with columns 'text' and 'label'")
    parser.add_argument("--model_output", required=True, help="Path to save final model (e.g., advanced_chunk_model.pt)")
    parser.add_argument("--output_details_csv", required=True, help="Path to save advanced evaluation details CSV")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per fold and for full training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for classifier")
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer (L2 regularization)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_augment", type=int, default=1, help="Number of augmented versions per example")
    parser.add_argument("--chunk_size", type=int, default=300, help="Words per chunk for embedding generation")
    parser.add_argument("--mc_iterations", type=int, default=10, help="Monte Carlo dropout iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda")
    args = parser.parse_args()
    train_and_evaluate(args)
