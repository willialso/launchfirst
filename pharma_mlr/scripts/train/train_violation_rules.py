#!/usr/bin/env python3
import os
import pandas as pd
import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from nltk.sentiment import SentimentIntensityAnalyzer

# === Paths ===
SCIBERT_PATH = "models/scibert_domain_finetuned/checkpoint-6115"
ROBERTA_PATH = "roberta-base"
MODEL_DIR = "models/violation_classifier"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Dataset ===
class CFRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Classifier ===
class CFRClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# === Embedding ===
def compute_embedding(texts, model_path, name):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for t in tqdm(texts, desc=name):
            inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)

def compute_sentiment(texts):
    sia = SentimentIntensityAnalyzer()
    return np.array([list(sia.polarity_scores(t).values()) for t in texts])  # shape (N, 4)

# === Main training ===
def main():
    df = pd.read_csv("violation_training_multilabel.csv")
    df.dropna(subset=["text", "fda_rules"], inplace=True)
    df["fda_rules"] = df["fda_rules"].apply(lambda x: x.split(",") if isinstance(x, str) else [])

    # 🎯 Labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["fda_rules"])
    joblib.dump(mlb, f"{MODEL_DIR}/cfr_label_binarizer.pkl")

    # 🧠 TFIDF + PCA (40)
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(df["text"]).toarray()
    pca = PCA(n_components=40)
    X_pca = pca.fit_transform(X_tfidf)
    joblib.dump(tfidf, f"{MODEL_DIR}/cfr_tfidf.pkl")
    joblib.dump(pca, f"{MODEL_DIR}/cfr_pca.pkl")

    # 🧠 Sentiment (4)
    X_sent = compute_sentiment(df["text"].tolist())

    # 🧠 SciBERT (768)
    X_scibert = compute_embedding(df["text"].tolist(), SCIBERT_PATH, "SciBERT")

    # 🧠 RoBERTa (768)
    X_roberta = compute_embedding(df["text"].tolist(), ROBERTA_PATH, "RoBERTa")

    # 🔀 Combine
    X_full = np.hstack([X_pca, X_sent, X_scibert, X_roberta])  # (N, 1580)
    assert X_full.shape[1] == 1580, f"Expected 1580 features, got {X_full.shape[1]}"

    # 🔃 Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    joblib.dump(scaler, f"{MODEL_DIR}/cfr_scaler.pkl")

    # 📚 Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_dl = DataLoader(CFRDataset(X_train, y_train), batch_size=8, shuffle=True)

    # 🏗️ Train MLP
    model = CFRClassifier(input_dim=1580, output_dim=y.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_dl):.4f}")

    # 💾 Save model
    torch.save(model.state_dict(), f"{MODEL_DIR}/cfr_model.pt")
    print(f"✅ Saved to {MODEL_DIR}/cfr_model.pt")

if __name__ == "__main__":
    main()
