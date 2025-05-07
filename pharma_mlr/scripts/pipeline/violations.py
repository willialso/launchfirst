# scripts/pipeline/violations.py
import os
import sys
import json
import logging
import joblib
import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scripts.utils.gcs_helpers import download_blob, upload_blob

# Environment variables with defaults
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlr_upload")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "/app/models")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (Updated to use environment variables)
SCIBERT_PATH = os.path.join(MODEL_BASE_PATH, "scibert_domain_finetuned")
ROBERTA_PATH = "roberta-base"
CHUNK_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "chunk_classifier/chunk_model.pt")
BINARIZER_PATH = os.path.join(MODEL_BASE_PATH, "chunk_classifier/label_binarizer.pkl")
TRANSFORMERS_PATH = os.path.join(MODEL_BASE_PATH, "chunk_classifier/feature_transformers.pkl")
VIOLATION_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "violation_detector")

class ChunkClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def load_artifacts():
    """Load all required model artifacts"""
    try:
        # Load chunk classifier artifacts
        label_binarizer = joblib.load(BINARIZER_PATH)
        checkpoint = torch.load(CHUNK_MODEL_PATH, map_location="cpu")
        model = ChunkClassifier(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        tfidf, pca = joblib.load(TRANSFORMERS_PATH)
        
        # Load violation detector
        violation_tokenizer = AutoTokenizer.from_pretrained(VIOLATION_MODEL_PATH)
        violation_model = AutoModelForSequenceClassification.from_pretrained(VIOLATION_MODEL_PATH)
        violation_pipe = pipeline(
            "text-classification",
            model=violation_model,
            tokenizer=violation_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return model, label_binarizer, tfidf, pca, violation_pipe
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        raise

def prepare_features(texts, tfidf, pca):
    """Prepare features using pre-fitted transformers"""
    try:
        scibert_model = SentenceTransformer(SCIBERT_PATH)
        roberta_model = SentenceTransformer(ROBERTA_PATH)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Get embeddings
        scibert_embeddings = scibert_model.encode(texts, convert_to_numpy=True)
        roberta_embeddings = roberta_model.encode(texts, convert_to_numpy=True)
        
        # Transform features
        X_tfidf = tfidf.transform(texts)
        X_pca = pca.transform(X_tfidf.toarray())
        
        # Get sentiment
        sentiments = np.array([
            list(sentiment_analyzer.polarity_scores(text).values())
            for text in texts
        ])
        
        # Combine features
        features = np.hstack([
            scibert_embeddings,
            roberta_embeddings,
            sentiments,
            X_pca
        ])
        
        return features
    except Exception as e:
        logger.error(f"Feature preparation failed: {str(e)}")
        raise

def evaluate_segments(file_id, segments_blob_path):
    """Main evaluation function"""
    try:
        logger.info(f"Starting evaluation for {file_id}")
        
        # Download segments
        local_segments = f"/tmp/{file_id}_segments.json"
        download_blob(BUCKET_NAME, segments_blob_path, local_segments)
        
        with open(local_segments) as f:
            segments = json.load(f).get("segments", [])
        
        texts = [seg.get("text", "") for seg in segments]
        
        # Load artifacts
        chunk_model, label_binarizer, tfidf, pca, violation_pipe = load_artifacts()
        
        # Prepare features for chunk classification
        X = prepare_features(texts, tfidf, pca)
        
        results = []
        csv_data = []
        
        for seg, x in zip(segments, X):
            # Chunk classification
            with torch.no_grad():
                input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                logits = chunk_model(input_tensor)
                probs = torch.sigmoid(logits).numpy()
                pred_idx = np.argmax(probs)
                predicted_label = label_binarizer.classes_[pred_idx]
                chunk_confidence = float(probs[0][pred_idx])
            
            # Violation detection (only run if chunk is classified as relevant)
            if predicted_label != "no_violation":
                violation_result = violation_pipe(seg.get("text", ""))
                violation_label = violation_result[0]["label"]
                violation_score = violation_result[0]["score"]
            else:
                violation_label = "no_violation"
                violation_score = 0.0
            
            results.append({
                "text": seg.get("text", ""),
                "page": seg.get("page", 1),
                "bbox": seg.get("bbox", []),
                "chunk_type": predicted_label,
                "chunk_confidence": chunk_confidence,
                "violation_label": violation_label,
                "violation_confidence": violation_score,
                "timestamp": seg.get("timestamp", "")
            })
            
            csv_data.append({
                "Text": seg.get("text", ""),
                "Page": seg.get("page", 1),
                "Chunk Type": predicted_label,
                "Chunk Confidence": chunk_confidence,
                "Violation Detected": violation_label,
                "Violation Confidence": violation_score
            })
        
        # Save outputs
        violations_path = f"/tmp/{file_id}_violations.json"
        verification_path = f"/tmp/{file_id}_verification.csv"
        
        with open(violations_path, "w") as f:
            json.dump({"violations": results}, f, indent=2)
        
        pd.DataFrame(csv_data).to_csv(verification_path, index=False)
        
        # Upload to GCS
        violations_blob = f"outputs/violations/{file_id}_violations.json"
        verification_blob = f"outputs/verification/{file_id}_verification.csv"
        
        upload_blob(BUCKET_NAME, violations_path, violations_blob)
        upload_blob(BUCKET_NAME, verification_path, verification_blob)
        
        return {
            "file_id": file_id,
            "status": "success",
            "violations_blob": violations_blob,
            "verification_csv": verification_blob,
            "violation_count": len([r for r in results if r["violation_label"] != "no_violation"])
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return {
            "file_id": file_id,
            "status": "error",
            "error": str(e)
        }

def main(file_id=None, segments_blob_path=None):
    if file_id is None:
        if len(sys.argv) != 3:
            logger.error("Usage: python violations.py <file_id> <segments_blob_path>")
            sys.exit(1)
        file_id = sys.argv[1]
        segments_blob_path = sys.argv[2]
    
    try:
        result = evaluate_segments(file_id, segments_blob_path)
        print(json.dumps(result))
        return result
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        error_result = {
            "file_id": file_id,
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(error_result))
        return error_result

if __name__ == "__main__":
    main()
