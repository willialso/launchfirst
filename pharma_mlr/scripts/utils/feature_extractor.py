# scripts/utils/feature_extractor.py (FINAL)

import numpy as np
import joblib
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModel

class FeatureExtractor:
    def __init__(self, model_paths: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_paths = model_paths

        # Correct: Load tfidf_vectorizer.pkl and tfidf_pca.pkl
        self.vectorizer = joblib.load(model_paths["tfidf_vectorizer"])  # tfidf_vectorizer.pkl
        self.pca = joblib.load(model_paths["tfidf_pca"])  # tfidf_pca.pkl

        self.sia = SentimentIntensityAnalyzer()

        self.scibert_tokenizer = AutoTokenizer.from_pretrained(model_paths["scibert"])
        self.scibert_model = AutoModel.from_pretrained(model_paths["scibert"])
        self.scibert_model.to(self.device).eval()

        self.roberta_tokenizer = AutoTokenizer.from_pretrained(model_paths["roberta"])
        self.roberta_model = AutoModel.from_pretrained(model_paths["roberta"])
        self.roberta_model.to(self.device).eval()

    def embed(self, text: str, model, tokenizer) -> np.ndarray:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).last_hidden_state
        return output.mean(dim=1).squeeze().cpu().numpy()

    def extract_sentiment(self, text: str) -> np.ndarray:
        scores = self.sia.polarity_scores(text)
        return np.array([scores["pos"], scores["neu"], scores["neg"], scores["compound"]])

    def prepare(self, text: str) -> np.ndarray:
        # 1. TF-IDF -> PCA
        tfidf = self.vectorizer.transform([text])
        tfidf_reduced = self.pca.transform(tfidf.toarray())[0]

        # 2. Sentiment (4 features)
        sentiment_features = self.extract_sentiment(text)

        # 3. Embeddings
        sci_emb = self.embed(text, self.scibert_model, self.scibert_tokenizer)
        rob_emb = self.embed(text, self.roberta_model, self.roberta_tokenizer)

        # 4. Combine all features (40 + 4 + 768 + 768 = 1580)
        return np.concatenate([tfidf_reduced, sentiment_features, sci_emb, rob_emb])
