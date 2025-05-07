#!/usr/bin/env python3
import os
import json
import time
import torch
import joblib
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler
from sentence_transformers         import SentenceTransformer, models

# ── Imports your MLP definition ───────────────────────────────────────────────
from scripts.models.model_architecture import MCDropoutMLP

# ── Paths & Hyperparams ──────────────────────────────────────────────────────
CSV_PATH     = "violation_training_multilabel.csv"
RULES_CONF   = "configs/violation_rules.json"
SCIBERT_DIR  = "models/scibert_domain_finetuned"
OUTPUT_DIR   = "models/violation_classifier"
TFIDF_MAX_F  = 5000
PCA_COMPONENTS = 50
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1) Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
with open(RULES_CONF) as f:
    RULE_KEYS = list(json.load(f).keys())
N_RULES = len(RULE_KEYS)

# Extract texts and multi-label arrays
texts = df["text"].tolist()
labels = []
for lst in df["fda_rules"].apply(json.loads):
    vec = [1 if RULE_KEYS[i] in lst else 0 for i in range(N_RULES)]
    labels.append(vec)
y = np.array(labels, dtype=np.float32)

# ── 2) Fit TF‑IDF → PCA ──────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_F)
X_t = tfidf.fit_transform(texts).toarray()

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_t)

# Save pipelines
joblib.dump(tfidf, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(pca,  os.path.join(OUTPUT_DIR, "pca.pkl"))

# ── 3) Load SentenceTransformers ─────────────────────────────────────────────
# SciBERT (domain-finetuned)
sc_model = models.Transformer(SCIBERT_DIR, max_seq_length=256)
sc_pool  = models.Pooling(sc_model.get_word_embedding_dimension())
scibert  = SentenceTransformer(modules=[sc_model, sc_pool])

# RoBERTa-base
roberta = SentenceTransformer("roberta-base")

# Encode texts (in batches to save memory)
def batch_encode(model, texts, batch_size=32):
    embs = []
    for i in range(0, len(texts), batch_size):
        embs.append(model.encode(texts[i : i+batch_size], show_progress_bar=False))
    return np.vstack(embs)

X_sc = batch_encode(scibert, texts, batch_size=16)
X_rb = batch_encode(roberta, texts, batch_size=16)

# ── 4) Build combined features ───────────────────────────────────────────────
# similarity feature: cosine between SciBERT & RoBERTa per sample
sims = np.array([
    np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)) 
    for a, b in zip(X_sc, X_rb)
]).reshape(-1,1)

X_all = np.hstack([X_sc, X_rb, sims, X_pca])

# ── 5) Standard scale ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# Prepare PyTorch dataset
tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32)
ds = TensorDataset(tensor_x, tensor_y)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# ── 6) Instantiate MLP ──────────────────────────────────────────────────────
model = MCDropoutMLP(input_dim=X_scaled.shape[1], hidden_dim=128, num_classes=N_RULES, dropout_p=0.5)
model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── 7) Training loop ─────────────────────────────────────────────────────────
print(f"Training on {len(ds)} samples, {N_RULES} rules → {X_scaled.shape[1]}‑dim features")
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bx.size(0)
    avg = total_loss / len(ds)
    print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg:.4f}")

# ── 8) Save the trained MLP ─────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "violation_mlp_multilabel.pt")
torch.save(model.state_dict(), out_path)
print(f"✅ Trained MLP saved to {out_path}")
