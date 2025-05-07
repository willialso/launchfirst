
#!/usr/bin/env python3
import os
import pandas as pd
import torch
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class CFRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CFRClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

def main():
    df = pd.read_csv("violation_training_multilabel.csv")
    df.dropna(subset=["text", "fda_rules"], inplace=True)
    df["fda_rules"] = df["fda_rules"].apply(lambda x: x.split(",") if isinstance(x, str) else [])

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["fda_rules"])
    joblib.dump(mlb, "models/violation_classifier/cfr_label_binarizer.pkl")

    tfidf = TfidfVectorizer(max_features=1000)
    pca = PCA(n_components=40)
    scaler = StandardScaler()

    X_tfidf = tfidf.fit_transform(df["text"]).toarray()
    X_pca = pca.fit_transform(X_tfidf)
    X_scaled = scaler.fit_transform(X_pca)

    joblib.dump(tfidf, "models/violation_classifier/cfr_tfidf.pkl")
    joblib.dump(pca, "models/violation_classifier/cfr_pca.pkl")
    joblib.dump(scaler, "models/violation_classifier/cfr_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    train_ds = CFRDataset(X_train, y_train)
    test_ds = CFRDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=8)

    model = CFRClassifier(input_dim=X_scaled.shape[1], output_dim=y.shape[1])
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
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dl):.4f}")

    torch.save(model.state_dict(), "models/violation_classifier/cfr_model.pt")

if __name__ == "__main__":
    main()
