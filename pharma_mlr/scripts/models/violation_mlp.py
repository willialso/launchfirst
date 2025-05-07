import numpy as np
import torch
import joblib
import time
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer, models
from scripts.models.model_architecture import MCDropoutMLP

# === Load Saved Components ===
scaler = joblib.load("models/violation_classifier/scaler.pkl")
vectorizer = joblib.load("models/violation_classifier/tfidf_vectorizer.pkl")
pca = joblib.load("models/violation_classifier/pca.pkl")
label_encoder = joblib.load("models/violation_classifier/label_encoder.pkl")
sia = joblib.load("models/violation_classifier/sia.pkl")

# === Load fine-tuned SciBERT ===
scibert_path = "/app/models/scibert_domain_finetuned"
scibert_model = models.Transformer(scibert_path, max_seq_length=256)
scibert_pooling = models.Pooling(scibert_model.get_word_embedding_dimension())
scibert = SentenceTransformer(modules=[scibert_model, scibert_pooling])

# === Load RoBERTa base ===
roberta = SentenceTransformer("roberta-base")

# === Load MLP model ===
model = MCDropoutMLP(input_dim=1580)
model.load_state_dict(torch.load("models/violation_classifier/violation_mlp_model.pt", map_location="cpu"))
model.eval()

# Enable dropout at inference for MC Dropout
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

enable_dropout(model)

# === Inference ===
def predict_violations(segments, threshold=0.2, num_mc_passes=10):
    texts = [s["text"] for s in segments]

    # Features
    X_tfidf = vectorizer.transform(texts).toarray()
    X_pca = pca.transform(X_tfidf)
    X_sentiment = np.array([list(sia.polarity_scores(t).values()) for t in texts])
    X_scibert = scibert.encode(texts, show_progress_bar=False)
    X_roberta = roberta.encode(texts, show_progress_bar=False)

    X_combined = np.concatenate([X_pca, X_sentiment, X_scibert, X_roberta], axis=1)
    X_scaled = scaler.transform(X_combined)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # MC Dropout
    all_preds = []
    for _ in range(num_mc_passes):
        with torch.no_grad():
            preds = model(X_tensor).squeeze().numpy()
            all_preds.append(preds)

    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)

    # LIME
    explainer = LimeTextExplainer()

    def predictor(text_batch):
        tfidf = vectorizer.transform(text_batch).toarray()
        pca_tf = pca.transform(tfidf)
        sent = np.array([list(sia.polarity_scores(t).values()) for t in text_batch])
        sbert = scibert.encode(text_batch, show_progress_bar=False)
        rbert = roberta.encode(text_batch, show_progress_bar=False)
        combo = np.concatenate([pca_tf, sent, sbert, rbert], axis=1)
        scaled = scaler.transform(combo)
        with torch.no_grad():
            return np.array([[p] for p in model(torch.tensor(scaled, dtype=torch.float32)).squeeze().numpy()])

    results = []
    for i, (segment, prob, uncertainty) in enumerate(zip(segments, mean_preds, std_preds)):
        pred_label = "violation" if prob > threshold else "no_violation"
        explanation = ""
        if pred_label == "violation" and prob > 0.6:
            start = time.time()
            lime_exp = explainer.explain_instance(segment["text"], predictor, num_features=2, num_samples=100, labels=[0])
            explanation = ", ".join([f"{w}: {w_val:.2f}" for w, w_val in lime_exp.as_list(label=0)])
            print(f"\u23F1\ufe0f LIME for segment {i} took {time.time() - start:.2f} sec")

        results.append({
            **segment,
            "prediction": pred_label,
            "confidence": round(float(prob), 3),
            "uncertainty": round(float(uncertainty), 3),
            "explanation": explanation
        })

    return results
