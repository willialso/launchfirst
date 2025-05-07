#!/usr/bin/env python3
"""
Advanced Master Pipeline

This script processes real OCR/segmentation outputs, runs advanced inference with four models,
and computes an overall weighted evaluation. There are two FDA evaluations:
  - Title 21 Evaluation (section-level rules) using the Title 21 overlay model.
  - FDA Bad Ad Evaluation (overall compliance) using a separate guidance JSON.

Models (in /app/models):
  • advanced_chunk_model.pt:
      Expects 1593 dims = 768 (embedding) + 825 (dummy features).
  • advanced_risk_benefit_model.pt:
      Expects 779 dims = 768 (embedding) + 10 (TF‑IDF PCA components) + 1 (sentiment).
  • advanced_blurry_line_model.pt:
      Expects raw 768‑dim embedding.
  • advanced_title21_overlay_model.pt:
      Expects 780 dims = 768 (embedding) + 10 (TF‑IDF PCA components) + 1 (sentiment) + 1 (normalized text length).

For risk–benefit and Title 21 extra features, the script computes:
   - Averaged embeddings from the SentenceTransformer (768 dims)
   - TF‑IDF features reduced via PCA (up to 10 components; padded to 10 if necessary)
   - VADER sentiment score (1 dim)
   - (For Title 21) Normalized text length (1 dim)
Thus:
   - Risk–benefit input: 768 + 10 + 1 = 779 dims.
   - Title 21 overlay input: 768 + 10 + 1 + 1 = 780 dims.

The Title 21 evaluation now loads a rules file (e.g., simplified_title21.json) that contains rules with a “key_term” for matching.
For each section, the script checks which rule key terms appear in the text and includes those as matched rules.

The script then aggregates section-level metrics using predefined layer and section weights.

File locations to check:
  - OCR outputs: GCS prefix provided by --promo_ocr_output_uri (e.g., gs://mlr_upload/ocr_output/)
  - Segmentation config: file at --seg_config (e.g., /app/configs/segmentation_config.json)
  - Title 21 rules: file at --title21_rules (e.g., /app/configs/simplified_title21.json)
  - FDA guidance: file at --bad_ad_guidance (e.g., /app/data/bad_ad/guidance.json)
  - Model checkpoints: in /app/models/
"""

import os, sys, argparse, logging, time, json, warnings
import torch, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import real OCR/Segmentation and FDA Evaluation Functions ---
from scripts.upload.promo_processor import (
    async_detect_document,
    load_config,
    advanced_segment_text,
    extract_metadata,
    generate_json_schema
)
from scripts.evaluate.bad_ad_evaluation import evaluate_bad_ad_compliance

# --- Function to load Title21 rules from a JSON file ---
def load_title21_rules(rules_path):
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_config = json.load(f)
            return rules_config.get("rules", [])
    else:
        logging.warning(f"Title21 rules file not found at {rules_path}.")
        return []

def match_title21_rules(text, rules):
    """Return a list of rules (titles or regulations) whose key_term appears in the text."""
    matches = []
    lower_text = text.lower()
    for rule in rules:
        key_term = rule.get("key_term", "").lower()
        if key_term and key_term in lower_text:
            matches.append(rule.get("regulation", "Unknown"))
    return matches if matches else ["No match"]

# --- OCR Download Function using google-cloud-storage ---
def download_ocr_output(gcs_output_uri, local_output_txt):
    if not gcs_output_uri.startswith("gs://"):
        logging.error("Invalid GCS URI: must start with gs://")
        return None
    parts = gcs_output_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        logging.error("No OCR output JSON files found in GCS at the specified prefix.")
        return None
    logging.info("Blobs found: " + ", ".join([blob.name for blob in blobs]))
    json_blobs = [blob for blob in blobs if blob.name.endswith(".json")]
    if not json_blobs:
        logging.error("No JSON file found in the blobs.")
        return None
    selected_blob = max(json_blobs, key=lambda b: b.updated)
    logging.info(f"Downloading OCR output from: {selected_blob.name}")
    if os.path.exists(local_output_txt):
        os.remove(local_output_txt)
    selected_blob.download_to_filename(local_output_txt)
    with open(local_output_txt, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def run_ocr_pipeline(gcs_input_uri, gcs_output_uri, local_output_txt):
    logging.info("Starting OCR processing...")
    async_detect_document(gcs_input_uri, gcs_output_uri)
    logging.info("Waiting for OCR operation to complete...")
    time.sleep(10)
    cleaned_text = download_ocr_output(gcs_output_uri, local_output_txt)
    if not cleaned_text:
        logging.error("OCR processing failed. Exiting.")
        return None
    logging.info("OCR processing completed.")
    return cleaned_text

def run_segmentation_pipeline(cleaned_text, seg_config, seg_output_folder, output_filename):
    logging.info(f"Loading segmentation configuration from {seg_config}")
    config = load_config(seg_config)
    if not config:
        logging.error("Segmentation configuration not loaded. Exiting.")
        return None
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    embedder_seg = SentenceTransformer("allenai/scibert_scivocab_uncased")
    segments = advanced_segment_text(cleaned_text, config, summarizer, embedder_seg)
    metadata = extract_metadata(cleaned_text, segments)
    schema = generate_json_schema(segments, cleaned_text, metadata)
    os.makedirs(seg_output_folder, exist_ok=True)
    output_json = os.path.join(seg_output_folder, output_filename)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    logging.info(f"Segmentation complete. JSON schema saved to {output_json}")
    return output_json

###################################
# Model Class Definitions
###################################

class AdvancedChunkClassifier(torch.nn.Module):
    def __init__(self, input_dim=1593, hidden_dim=64):
        super(AdvancedChunkClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class AdvancedBenefitRiskClassifier(torch.nn.Module):
    def __init__(self, input_dim=779, hidden_dim=128):
        super(AdvancedBenefitRiskClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu3 = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_dim // 2, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return self.out(x)

class AdvancedBlurryLineClassifier(torch.nn.Module):
    def __init__(self, input_dim=768):
        super(AdvancedBlurryLineClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.linear(x)

class AdvancedTitle21OverlayClassifier(torch.nn.Module):
    def __init__(self, input_dim=780, hidden_dim=128):
        super(AdvancedTitle21OverlayClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu3 = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden_dim // 2, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return self.out(x)

###################################
# Inference Functions
###################################

def run_advanced_chunk_inference(schema, embedder, device):
    preds = {}
    try:
        model = AdvancedChunkClassifier(input_dim=1593, hidden_dim=64)
        model_path = os.path.join(project_root, "models", "advanced_chunk_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        for section in schema.get("sections", []):
            text = section.get("text", "").strip()
            if not text:
                continue
            emb = embedder.encode([text])[0]
            dummy_feats = np.ones(825) * 0.5
            full_feat = np.concatenate([emb, dummy_feats])
            input_tensor = torch.tensor(full_feat, dtype=torch.float).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            preds[section.get("section_type", "UNKNOWN")] = {"text": text, "chunk_prediction": probs}
        logging.info("Advanced chunk inference completed.")
    except Exception as e:
        logging.error("Advanced chunk inference error: " + str(e))
    return preds

def run_advanced_blurry_inference(schema, embedder, device):
    preds = {}
    try:
        model = AdvancedBlurryLineClassifier(input_dim=768)
        model_path = os.path.join(project_root, "models", "advanced_blurry_line_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        for section in schema.get("sections", []):
            text = section.get("text", "").strip()
            if not text:
                continue
            emb = embedder.encode([text])[0]
            input_tensor = torch.tensor(emb, dtype=torch.float).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            preds[section.get("section_type", "UNKNOWN")] = {"text": text, "blurry_prediction": probs}
        logging.info("Advanced blurry inference completed.")
    except Exception as e:
        logging.error("Advanced blurry inference error: " + str(e))
    return preds

def compute_real_extra_features(schema, embedder):
    texts = [section.get("text", "").strip() for section in schema.get("sections", []) if section.get("text", "").strip()]
    if not texts:
        logging.error("No section texts found for feature extraction.")
        return None, None, []
    avg_emb = embedder.encode(texts)  # (N,768)
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    n_components = min(10, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    tfidf_feats = pca.fit_transform(tfidf_matrix.toarray())  # (N, n_components)
    desired = 10
    if tfidf_feats.shape[1] < desired:
        pad_width = desired - tfidf_feats.shape[1]
        pad = np.zeros((tfidf_feats.shape[0], pad_width))
        tfidf_feats = np.concatenate([tfidf_feats, pad], axis=1)
    analyzer = SentimentIntensityAnalyzer()
    sentiment = np.array([analyzer.polarity_scores(t)["compound"] for t in texts]).reshape(-1,1)
    lengths = np.array([len(t) for t in texts]).reshape(-1,1)
    norm_length = lengths / lengths.max()
    extra_features_rb = np.concatenate([avg_emb, tfidf_feats, sentiment], axis=1)  # 768+10+1 = 779
    extra_features_title21 = np.concatenate([avg_emb, tfidf_feats, sentiment, norm_length], axis=1)  # 768+10+1+1 = 780
    return extra_features_rb, extra_features_title21, texts

def run_real_risk_benefit_inference(schema, extra_features, device):
    preds = {}
    try:
        model = AdvancedBenefitRiskClassifier(input_dim=779, hidden_dim=128)
        model_path = os.path.join(project_root, "models", "advanced_risk_benefit_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        idx = 0
        for section in schema.get("sections", []):
            text = section.get("text", "").strip()
            if not text:
                continue
            feat = extra_features[idx]
            input_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)
            logits = model(input_tensor)
            risk_value = torch.sigmoid(logits).squeeze().item()
            preds[section.get("section_type", "UNKNOWN")] = {"text": text, "risk_benefit_prediction": [risk_value, 1 - risk_value]}
            idx += 1
        logging.info("Advanced risk–benefit inference (real features) completed.")
    except Exception as e:
        logging.error("Advanced risk–benefit inference error: " + str(e))
    return preds

def run_real_title21_inference(schema, extra_features, device, title21_rules):
    preds = {}
    try:
        model = AdvancedTitle21OverlayClassifier(input_dim=780, hidden_dim=128)
        model_path = os.path.join(project_root, "models", "advanced_title21_overlay_model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        idx = 0
        for section in schema.get("sections", []):
            text = section.get("text", "").strip()
            if not text:
                continue
            feat = extra_features[idx]
            input_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            # Use the Title21 rules to match key terms in the text
            matched_rules = match_title21_rules(text, title21_rules)
            preds[section.get("section_type", "UNKNOWN")] = {"text": text, "title21_prediction": probs, "matched_rules": matched_rules}
            idx += 1
        logging.info("Advanced Title21 overlay inference (real features) completed.")
    except Exception as e:
        logging.error("Advanced Title21 inference error: " + str(e))
    return preds

def load_title21_rules(rules_path):
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_config = json.load(f)
            return rules_config.get("rules", [])
    else:
        logging.warning(f"Title21 rules file not found at {rules_path}.")
        return []

def match_title21_rules(text, rules):
    matches = []
    lower_text = text.lower()
    for rule in rules:
        key_term = rule.get("key_term", "").lower()
        if key_term and key_term in lower_text:
            matches.append(rule.get("regulation", "Unknown"))
    return matches if matches else ["No match"]

###################################
# Aggregation & Output
###################################
def main():
    parser = argparse.ArgumentParser(description="Advanced Master Pipeline")
    parser.add_argument("--promo_gcs_uri", required=True, help="GCS URI of the promo PDF")
    parser.add_argument("--promo_ocr_output_uri", required=True, help="GCS URI for promo OCR output")
    parser.add_argument("--local_ocr_txt", default="promo_ocr.txt", help="Local file to save promo OCR text")
    parser.add_argument("--seg_config", required=True, help="Path to segmentation config JSON")
    parser.add_argument("--seg_output_folder", required=True, help="Folder to save segmentation JSON")
    parser.add_argument("--pi_gcs_uri", help="GCS URI of the PI PDF (optional)")
    parser.add_argument("--pi_ocr_output_uri", help="GCS URI for PI OCR output (optional)")
    parser.add_argument("--local_pi_ocr_txt", default="pi_ocr.txt", help="Local file to save PI OCR text")
    parser.add_argument("--granular_output_csv", required=True, help="Path to save advanced evaluation CSV")
    parser.add_argument("--bad_ad_guidance", required=True, help="Path to FDA Bad Ad guidance JSON")
    parser.add_argument("--title21_rules", required=True, help="Path to Title21 rules JSON (e.g., simplified_title21.json)")
    args = parser.parse_args()

    promo_text = run_ocr_pipeline(args.promo_gcs_uri, args.promo_ocr_output_uri, args.local_ocr_txt)
    if not promo_text:
        logging.error("Promo OCR failed. Exiting pipeline.")
        return
    promo_json_path = run_segmentation_pipeline(promo_text, args.seg_config, args.seg_output_folder, "promo_mapping.json")
    if not promo_json_path:
        logging.error("Promo segmentation failed. Exiting pipeline.")
        return

    pi_json_path = None
    if args.pi_gcs_uri and args.pi_ocr_output_uri:
        pi_text = run_ocr_pipeline(args.pi_gcs_uri, args.pi_ocr_output_uri, args.local_pi_ocr_txt)
        if not pi_text:
            logging.error("PI OCR failed. Proceeding with promo-only evaluation.")
        else:
            pi_json_path = run_segmentation_pipeline(pi_text, args.seg_config, args.seg_output_folder, "pi_mapping.json")
            logging.info(f"PI segmentation complete. JSON schema saved to {pi_json_path}")
    else:
        logging.info("No PI provided; proceeding with promo-only evaluation.")

    promo_schema_path = os.path.join(args.seg_output_folder, "promo_mapping.json")
    with open(promo_schema_path, "r", encoding="utf-8") as f:
        promo_schema = json.load(f)

    embedder = SentenceTransformer("allenai/scibert_scivocab_uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    chunk_results = run_advanced_chunk_inference(promo_schema, embedder, device)
    blurry_results = run_advanced_blurry_inference(promo_schema, embedder, device)

    extra_features_rb, extra_features_title21, texts_order = compute_real_extra_features(promo_schema, embedder)
    if extra_features_rb is None or extra_features_title21 is None:
        logging.error("Failed to compute extra features for real evaluation.")
        return

    rb_results = run_real_risk_benefit_inference(promo_schema, extra_features_rb, device)
    # Load Title21 rules from file
    title21_rules = load_title21_rules(args.title21_rules)
    title21_results = run_real_title21_inference(promo_schema, extra_features_title21, device, title21_rules)

    if os.path.exists(args.bad_ad_guidance):
        with open(args.bad_ad_guidance, "r") as f:
            guidance = json.load(f)
    else:
        logging.warning(f"Guidance file not found at {args.bad_ad_guidance}. Using default guidance.")
        guidance = {"fda_keywords": ["violation", "noncompliant", "FDA", "rule breach"]}

    fda_bad_ad_results = evaluate_bad_ad_compliance(promo_schema, args.bad_ad_guidance, threshold=0.3)
    fda_bad_ad_percent = fda_bad_ad_results.get("overall_compliance", 0) * 100 if fda_bad_ad_results.get("overall_compliance") is not None else 0

    section_weights = {
        "EFFICACY_CLAIM": 0.40,
        "IMPORTANT_SAFETY_INFORMATION": 0.35,
        "INDICATION": 0.20,
        "PROMO_HEADLINE": 0.10,
        "REFERENCES": 0.10
    }
    layer_weights = {
        "blurry": 0.40,
        "fda": 0.30,
        "risk_benefit": 0.20,
        "fda_bad_ad": 0.10
    }

    granular_rows = []
    section_grades = []
    for section in promo_schema.get("sections", []):
        text = section.get("text", "").strip()
        if not text:
            continue
        key = section.get("section_type", "UNKNOWN")
        chunk_pred = chunk_results.get(key, {}).get("chunk_prediction", [0.5, 0.5])
        rb_pred = rb_results.get(key, {}).get("risk_benefit_prediction", [0.5, 0.5])
        blurry_pred = blurry_results.get(key, {}).get("blurry_prediction", [0.5, 0.5])
        title21_pred = title21_results.get(key, {}).get("title21_prediction", [0.5, 0.5])
        
        blurry_percent = blurry_pred[1] * 100
        fda_percent = 100 if any(kw.lower() in text.lower() for kw in guidance.get("fda_keywords", [])) else 0
        risk_percent = rb_pred[0] * 100
        fda_bad_ad = fda_bad_ad_percent
        
        layer_grade = (layer_weights["blurry"] * blurry_percent +
                       layer_weights["fda"] * fda_percent +
                       layer_weights["risk_benefit"] * risk_percent +
                       layer_weights["fda_bad_ad"] * fda_bad_ad)
        sec_weight = section_weights.get(key, 1)
        section_grade = layer_grade * sec_weight
        section_grades.append(section_grade)
        
        chunk_pass = chunk_pred[1] * 100
        title21_pass = title21_pred[1] * 100
        
        granular_rows.append({
            "section_type": key,
            "promo_text": text,
            "chunk_pass_percent": chunk_pass,
            "fda_compliant_percent": fda_percent,
            "blurry_line_percent": blurry_percent,
            "title21_pass_percent": title21_pass,
            "risk_percent": risk_percent,
            "fda_bad_ad_percent": fda_bad_ad,
            "layer_grade": layer_grade,
            "section_weight": sec_weight,
            "section_grade": section_grade,
            "matched_rules": title21_results.get(key, {}).get("matched_rules", [])
        })
    
    overall_grade = np.mean(section_grades) if section_grades else None

    aggregated = {
        "avg_chunk_pass_percent": np.mean([row["chunk_pass_percent"] for row in granular_rows]) if granular_rows else None,
        "avg_fda_compliant_percent": np.mean([row["fda_compliant_percent"] for row in granular_rows]) if granular_rows else None,
        "avg_blurry_line_percent": np.mean([row["blurry_line_percent"] for row in granular_rows]) if granular_rows else None,
        "avg_title21_pass_percent": np.mean([row["title21_pass_percent"] for row in granular_rows]) if granular_rows else None,
        "avg_risk_percent": np.mean([row["risk_percent"] for row in granular_rows]) if granular_rows else None,
        "avg_fda_bad_ad_percent": np.mean([row["fda_bad_ad_percent"] for row in granular_rows]) if granular_rows else None,
        "overall_grade": overall_grade
    }
    
    logging.info("Aggregated advanced evaluation metrics:")
    logging.info(json.dumps(aggregated, indent=2))
    
    agg_output_file = os.path.join(args.seg_output_folder, "aggregated_advanced_metrics.json")
    with open(agg_output_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    logging.info(f"Aggregated advanced metrics saved to {agg_output_file}")
    
    df_granular = pd.DataFrame(granular_rows)
    df_granular.to_csv(args.granular_output_csv, index=False)
    logging.info(f"Granular advanced evaluation details saved to {args.granular_output_csv}")
    
    summary = df_granular.groupby("section_type").agg({
        "chunk_pass_percent": "mean",
        "fda_compliant_percent": "mean",
        "blurry_line_percent": "mean",
        "title21_pass_percent": "mean",
        "risk_percent": "mean",
        "layer_grade": "mean",
        "section_grade": "mean"
    }).reset_index()
    logging.info("Section-level summary:")
    logging.info(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.seg_output_folder, "section_summary.csv"), index=False)
    
    logging.info("Advanced master pipeline processing complete.")

if __name__ == "__main__":
    main()
