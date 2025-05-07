# scripts/models/chunk_classifier_model.py

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Paths
SCIBERT_PATH = "models/scibert_domain_finetuned"
ROBERTA_PATH = "roberta-base"

class ChunkClassifier(nn.Module):
    def __init__(self, input_dim=1536, output_dim=7):
        super(ChunkClassifier, self).__init__()
        self.net = nn.Sequential(  # Changed from fc to net to match saved model
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)  # Changed from fc to net

def prepare_features(texts):
    """Extract SciBERT and RoBERTa embeddings and concatenate them"""
    # Initialize models and tokenizers
    scibert_tokenizer = AutoTokenizer.from_pretrained(SCIBERT_PATH)
    scibert_model = AutoModel.from_pretrained(SCIBERT_PATH)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
    roberta_model = AutoModel.from_pretrained(ROBERTA_PATH)

    # Set models to evaluation mode
    scibert_model.eval()
    roberta_model.eval()

    features = []
    for text in texts:
        # Tokenize and get embeddings from SciBERT
        scibert_inputs = scibert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Tokenize and get embeddings from RoBERTa
        roberta_inputs = roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            scibert_outputs = scibert_model(**scibert_inputs)
            roberta_outputs = roberta_model(**roberta_inputs)

        # Pool the embeddings (mean pooling)
        scibert_emb = scibert_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        roberta_emb = roberta_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # Concatenate the embeddings
        combined = np.concatenate([scibert_emb, roberta_emb])
        features.append(combined)

    return np.array(features)

# Optional: Add model verification utility
def verify_model_loading(model_path, input_dim=1536, output_dim=7):
    """Utility function to verify model can be loaded properly"""
    model = ChunkClassifier(input_dim, output_dim)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Example usage for testing
if __name__ == "__main__":
    # Test the model architecture
    test_model = ChunkClassifier()
    print("Model architecture:")
    print(test_model)
    
    # Test feature preparation
    test_texts = ["This is a test sentence.", "Another example text."]
    print("\nTesting feature preparation...")
    features = prepare_features(test_texts)
    print(f"Features shape: {features.shape}")
    
    # Verify model loading (uncomment and set correct path to test)
    # print("\nVerifying model loading...")
    # verify_model_loading("models/chunk_classifier/chunk_model.pt")
