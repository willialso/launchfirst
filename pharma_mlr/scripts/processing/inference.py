#!/usr/bin/env python3
"""
Inference Processor Script for Promo Segmentation

This script performs the following:
  1. Triggers OCR on a promo PDF via the Google Vision API using functions from promo_processor.py.
  2. Downloads the raw OCR JSON output from GCS.
  3. Extracts bounding box data (text and numeric features) from the OCR JSON.
  4. Computes text embeddings for each extracted region using SentenceTransformer (SciBERT).
  5. Loads a trained segmentation model (model.pt) and a segmentation config file (bbox_segmentation_config.json)
     that maps model outputs to human-readable labels.
  6. Runs inference to predict a segmentation label for each bounding box.
  7. Returns a list of dictionaries for each region (including source_pdf, text, bounding box coordinates,
     area, prominence, and predicted_label).
     
Usage Example:
  python processing/inference.py \
    --pdf_uri gs://mlr_upload/promo/YourPromo.pdf \
    --gcs_output_uri gs://mlr_upload/ocr_output/ \
    --ocr_wait_time 15 \
    --model_path model.pt \
    --config_path configs/bbox_segmentation_config.json \
    --output_json outputs/segmentation_results.json
"""

import argparse
import json
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Import OCR functions from promo_processor.py (make sure these are defined there)
from processing.promo_processor import async_detect_document, download_ocr_output

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Local helper functions for extraction
# ---------------------------
def compute_prominence(x_min, y_min, x_max, y_max):
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    center_x = (x_min + x_max) / 2.0
    horizontal_factor = 1 - 2 * abs(center_x - 0.5)
    prominence = (0.5 * height + 0.5 * area) * (1 - y_min) * horizontal_factor
    return prominence, area

def extract_paragraph_bbox(page):
    """Extract bounding box data for each paragraph on a given page."""
    rows = []
    for block in page.get("blocks", []):
        for paragraph in block.get("paragraphs", []):
            text = ""
            vertices = []
            for word in paragraph.get("words", []):
                word_text = "".join(symbol.get("text", "") for symbol in word.get("symbols", []))
                text += word_text + " "
                bbox = word.get("boundingBox", {})
                v = bbox.get("normalizedVertices") or bbox.get("vertices") or []
                vertices.extend(v)
            text = text.strip()
            if len(text) < 5:
                continue
            xs = [v.get("x", 0) for v in vertices if "x" in v]
            ys = [v.get("y", 0) for v in vertices if "y" in v]
            if not xs or not ys:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            prominence, area = compute_prominence(x_min, y_min, x_max, y_max)
            row = {
                "source_pdf": None,
                "text": text,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "area": area,
                "prominence": prominence
            }
            rows.append(row)
    return rows

def extract_bbox_data(ocr_response):
    """Extract bounding box data from OCR JSON for all pages."""
    rows = []
    for res in ocr_response.get("responses", []):
        full_annotation = res.get("fullTextAnnotation", {})
        pages = full_annotation.get("pages", [])
        for page in pages:
            page_rows = extract_paragraph_bbox(page)
            rows.extend(page_rows)
    return rows

def download_raw_ocr_json(gcs_output_uri):
    """
    Downloads raw OCR JSON responses from the specified GCS output URI.
    Returns a dictionary with a "responses" key containing the merged OCR responses.
    """
    from google.cloud import storage
    storage_client = storage.Client()
    if not gcs_output_uri.startswith("gs://"):
        raise ValueError("GCS output URI must start with gs://")
    parts = gcs_output_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    bucket = storage_client.bucket(bucket_name)
    blobs = [blob for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith(".json")]
    if not blobs:
        logging.error("No OCR JSON files found in GCS at the specified prefix.")
        return None
    blobs.sort(key=lambda b: b.name)
    full_response = {"responses": []}
    for blob in blobs:
        logging.info(f"Processing blob: {blob.name}")
        try:
            content = blob.download_as_string().decode("utf-8", errors="replace")
            response = json.loads(content)
            full_response["responses"].extend(response.get("responses", []))
        except Exception as e:
            logging.error(f"Error processing blob {blob.name}: {e}")
    return full_response

# ---------------------------
# Model Definition (must match your training script)
# ---------------------------
class CombinedClassifier(nn.Module):
    def __init__(self, embedding_dim, numeric_dim, num_classes):
        super(CombinedClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, emb, numeric):
        x = torch.cat([emb, numeric], dim=1)
        return self.fc(x)

def load_segmentation_config(config_path):
    """
    Loads segmentation config from a JSON file.
    Expected format:
    {
      "label_map": {
        "0": "GENERAL",
        "1": "IMPORTANT_SAFETY_INFORMATION",
        "2": "INDICATIONS",
        "3": "INSTRUCTIONS",
        "4": "EFFICACY_CLAIMS",
        "5": "PROMO_HEADLINE",
        "6": "DISCLAIMER",
        "7": "REFERENCE",
        "8": "UNKNOWN"
      }
    }
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("label_map", {})

def run_inference(pdf_uri, gcs_output_uri, wait_time, model_path, config_path):
    """
    Runs OCR on the promo PDF, extracts bounding box regions, computes text embeddings,
    loads the trained segmentation model, and predicts segmentation labels.
    
    Returns a list of dictionaries for each region with:
      - source_pdf, text, x_min, y_min, x_max, y_max, area, prominence, predicted_label
    """
    logging.info("Starting OCR inference on promo PDF...")
    async_detect_document(pdf_uri, gcs_output_uri)
    logging.info(f"Waiting {wait_time} seconds for OCR output...")
    time.sleep(wait_time)
    
    raw_ocr_json = download_raw_ocr_json(gcs_output_uri)
    if not raw_ocr_json:
        logging.error("No OCR JSON data extracted.")
        return None
    
    regions = extract_bbox_data(raw_ocr_json)
    if not regions:
        logging.error("No regions extracted from OCR data.")
        return None
    
    for region in regions:
        region["source_pdf"] = pdf_uri

    texts = [region["text"] for region in regions]
    numeric_features = [
        [region["x_min"], region["y_min"], region["x_max"], region["y_max"], region["area"], region["prominence"]]
        for region in regions
    ]
    numeric_features = torch.tensor(numeric_features, dtype=torch.float32).to(device)

    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased", device=device)
    text_embeddings = embedder.encode(texts, convert_to_tensor=True).to(device)

    label_map = load_segmentation_config(config_path)
    num_classes = len(label_map)
    embedding_dim = text_embeddings.shape[1]
    numeric_dim = numeric_features.shape[1]

    model = CombinedClassifier(embedding_dim, numeric_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(text_embeddings, numeric_features)
        _, predicted_indices = torch.max(outputs, dim=1)
    predicted_indices = predicted_indices.cpu().numpy()

    predictions = []
    for region, pred in zip(regions, predicted_indices):
        pred_label = label_map.get(str(pred), "UNKNOWN")
        region["predicted_label"] = pred_label
        predictions.append(region)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on a promo PDF and output predictions as JSON.")
    parser.add_argument("--pdf_uri", required=True,
                        help="GCS URI of the promo PDF (e.g., gs://mlr_upload/promo/YourPromo.pdf)")
    parser.add_argument("--gcs_output_uri", required=True,
                        help="GCS URI for OCR output (e.g., gs://mlr_upload/ocr_output/)")
    parser.add_argument("--ocr_wait_time", type=int, default=15,
                        help="Time to wait after OCR request (in seconds)")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained segmentation model (e.g., model.pt)")
    parser.add_argument("--config_path", required=True,
                        help="Path to segmentation config file (e.g., configs/bbox_segmentation_config.json)")
    parser.add_argument("--output_json", required=True,
                        help="Path to save segmentation predictions as JSON (e.g., outputs/segmentation_results.json)")
    args = parser.parse_args()

    predictions = run_inference(args.pdf_uri, args.gcs_output_uri, args.ocr_wait_time, args.model_path, args.config_path)
    if predictions is None:
        logging.error("Inference failed. No predictions generated.")
        return
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"Segmentation predictions saved to {args.output_json}")

if __name__ == "__main__":
    main()
