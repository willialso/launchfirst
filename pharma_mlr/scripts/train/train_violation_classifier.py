import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load dataset
df = pd.read_csv("data/violation_training_dataset.csv")
df = df[df["section_type"] == "EFFICACY_CLAIM"]
df = df.dropna(subset=["text", "violation_label"])

# Binary label
df["violation_binary"] = df["violation_label"].apply(
    lambda x: "violation" if str(x).strip().lower() != "no_violation" else "no_violation"
)

X = df["text"]
y = df["violation_binary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline with TF-IDF, SMOTE, Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Save model
os.makedirs("models/violation_classifier", exist_ok=True)
joblib.dump(pipeline, "models/violation_classifier/violation_classifier.pkl")
print("✅ Model saved to: models/violation_classifier/violation_classifier.pkl")
