import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
import joblib
import nltk

nltk.download("vader_lexicon")

# Load data
df = pd.read_csv("data/violation_training_dataset.csv")
df = df[df["text"].notnull() & df["violation_label"].notnull()]

# Binary label: violation or no_violation
df["label"] = df["violation_label"].apply(
    lambda x: "no_violation" if str(x).strip() == "no_violation" else "violation"
)

# Label encoding
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])  # 0 = no_violation, 1 = violation

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df["text"]).toarray()

# PCA on TF-IDF
max_components = min(40, X_tfidf.shape[0], X_tfidf.shape[1])
pca = PCA(n_components=max_components)
X_pca = pca.fit_transform(X_tfidf)

# Sentiment features
sia = SentimentIntensityAnalyzer()
sentiment_features = df["text"].apply(lambda x: pd.Series(sia.polarity_scores(x)))
X_sentiment = sentiment_features.to_numpy()

# SciBERT Embeddings
model = SentenceTransformer("allenai/scibert_scivocab_uncased")
X_bert = model.encode(df["text"].tolist(), show_progress_bar=True)

# Combine all features
X_full = np.concatenate([X_pca, X_sentiment, X_bert], axis=1)
y = df["label_encoded"]

# Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_full, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Classifier with early stopping
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    early_stopping=True,
    random_state=42,
    max_iter=200
)
clf.fit(X_train_scaled, y_train)

# Evaluation
y_pred = clf.predict(X_test_scaled)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model + components
os.makedirs("models/violation_classifier", exist_ok=True)
joblib.dump(clf, "models/violation_classifier/violation_mlp_model.pkl")
joblib.dump(scaler, "models/violation_classifier/scaler.pkl")
joblib.dump(vectorizer, "models/violation_classifier/tfidf_vectorizer.pkl")
joblib.dump(pca, "models/violation_classifier/pca.pkl")
joblib.dump(le, "models/violation_classifier/label_encoder.pkl")

print("\n✅ Violation MLP model saved.")
