import os
import torch
import joblib
from typing import List, Dict
from scripts.utils.feature_extractor import FeatureExtractor
from scripts.models.model_architecture import MCDropoutMLP

class ViolationDetector:
    def __init__(self, model_paths: Dict[str, str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_paths = model_paths
        self.load_models()
        self.feature_extractor = FeatureExtractor(model_paths)

    def load_models(self):
        self.violation_model = MCDropoutMLP(input_dim=1580)
        state_dict = torch.load(self.model_paths['model'], map_location=self.device)
        self.violation_model.load_state_dict(state_dict)
        self.violation_model = self.violation_model.to(self.device).eval()

    def predict_violation(self, segment: Dict) -> Dict:
        features = self.feature_extractor.prepare(segment['text'])
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.violation_model(features_tensor).item()

        return {
            'text': segment['text'],
            'page': segment.get('page', 1),
            'bbox': segment.get('bbox'),
            'tokens': segment.get('tokens'),
            'prediction': 'violation' if score > 0.5 else 'no_violation',
            'confidence': float(score),
            'rule': segment.get('rule', 'unknown')
        }

def load_default_detector():
    model_paths = {
        'model': '/app/models/violation_classifier/violation_mlp_model.pt',
        'tfidf_pca': '/app/models/violation_classifier/tfidf_pca_vectorizer.joblib',
        'scibert': '/app/models/scibert_domain_finetuned',
        'roberta': 'roberta-base'
    }
    return ViolationDetector(model_paths)

def evaluate_segments_with_violation_model(file_id: str, segments: List[Dict]) -> List[Dict]:
    detector = load_default_detector()
    results = []
    for segment in segments:
        if isinstance(segment, str):
            segment = {"text": segment}
        elif not isinstance(segment, dict) or 'text' not in segment:
            continue
        result = detector.predict_violation(segment)
        results.append(result)
    return results
