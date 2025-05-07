# scripts/models/cfr_model.py

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

SCIBERT_PATH = "models/scibert_domain_finetuned"
ROBERTA_PATH = "roberta-base"

class CFRClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 💥 ADD THIS BELOW
# Feature extractor function
def prepare_features(texts):
    scibert_tokenizer = AutoTokenizer.from_pretrained(SCIBERT_PATH)
    scibert_model = AutoModel.from_pretrained(SCIBERT_PATH)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    roberta_model = AutoModel.from_pretrained(ROBERTA_PATH)

    features = []
    for text in texts:
        scibert_inputs = scibert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        roberta_inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            scibert_outputs = scibert_model(**scibert_inputs)
            roberta_outputs = roberta_model(**roberta_inputs)

        scibert_emb = scibert_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        roberta_emb = roberta_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        combined = np.concatenate([scibert_emb, roberta_emb])
        features.append(combined)

    return np.array(features)
