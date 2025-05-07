# === ✅ PyTorch Training Script for Violation MLP Classifier (1580-dim input) ===

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer, models
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

nltk.download("vader_lexicon")

# === Load and Preprocess Data ===
df = pd.read_csv("/app/data/violation_training_dataset.csv")
df = df[df["text"].notnull() & df["violation_label"].notnull()]
df["label"] = df["violation_label"].apply(lambda x: "no_violation" if str(x).strip() == "no_violation" else "violation")
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# === Feature Extraction ===
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["text"]).toarray()

max_components = min(40, X_tfidf.shape[0], X_tfidf.shape[1])
pca = PCA(n_components=max_components)
X_pca = pca.fit_transform(X_tfidf)

sia = SentimentIntensityAnalyzer()
sentiment_features = df["text"].apply(lambda x: pd.Series(sia.polarity_scores(x)))
X_sentiment = sentiment_features.to_numpy()

# ✅ SciBERT (fine-tuned)
scibert_path = "/app/models/scibert_domain_finetuned"
scibert_transformer = models.Transformer(scibert_path, max_seq_length=256)
scibert_pooling = models.Pooling(scibert_transformer.get_word_embedding_dimension())
scibert = SentenceTransformer(modules=[scibert_transformer, scibert_pooling])
X_scibert = scibert.encode(df["text"].tolist(), show_progress_bar=True)

# ✅ RoBERTa (base model)
roberta = SentenceTransformer("roberta-base")
X_roberta = roberta.encode(df["text"].tolist(), show_progress_bar=True)

# === Combine Features ===
X_full = np.concatenate([X_pca, X_sentiment, X_scibert, X_roberta], axis=1)
print("✅ Final input dimension:", X_full.shape[1])
y = df["label_encoded"]

# === Oversampling and Scaling ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_full, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === PyTorch Dataset and Loader ===
X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === Define Dropout MLP ===
class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# === Initialize and Train ===
model = MCDropoutMLP(X_train_scaled.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

model.train()
for epoch in range(20):
    epoch_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds_test = model(torch.tensor(X_test_scaled, dtype=torch.float32)).squeeze()
    y_pred = (preds_test >= 0.5).int().numpy()

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save Everything ===
output_dir = "/app/models/violation_classifier"
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(output_dir, "violation_mlp_model.pt"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
joblib.dump(pca, os.path.join(output_dir, "pca.pkl"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

print("\n✅ PyTorch MLP with dropout saved.")
