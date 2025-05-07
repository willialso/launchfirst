# scripts/pipeline/annotate.py
import os
import sys
import json
import fitz  # PyMuPDF
import logging
from scripts.utils.gcs_helpers import download_blob, upload_blob

# Environment variables with defaults
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlr_upload")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "/app/models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Main Annotate Function ===
def annotate_pdf(file_id: str, violations_blob: str):
    try:
        logger.info(f"[ANNOTATE] Starting annotation for {file_id}")

        # === Step 1: Download PDF and violations ===
        local_pdf_path = f"/tmp/{file_id}.pdf"
        local_violations_path = f"/tmp/{file_id}_violations.json"

        download_blob(BUCKET_NAME, f"promo_uploads/{file_id}.pdf", local_pdf_path)
        download_blob(BUCKET_NAME, violations_blob, local_violations_path)

        with open(local_violations_path, "r") as f:
            violations_data = json.load(f)
        
        segments = violations_data.get("violations", [])
        if not segments:
            logger.warning(f"[ANNOTATE] No violations to annotate for {file_id}")

        # === Step 2: Open PDF ===
        doc = fitz.open(local_pdf_path)
        tooltips = []

        for seg in segments:
            page_num = seg.get("page", 1) - 1  # 0-indexed
            bbox = seg.get("bbox")
            
            if bbox and 0 <= page_num < len(doc):
                page = doc[page_num]
                rect = fitz.Quad(
                    fitz.Point(bbox[0]["x"] * page.rect.width, bbox[0]["y"] * page.rect.height),
                    fitz.Point(bbox[1]["x"] * page.rect.width, bbox[1]["y"] * page.rect.height),
                    fitz.Point(bbox[2]["x"] * page.rect.width, bbox[2]["y"] * page.rect.height),
                    fitz.Point(bbox[3]["x"] * page.rect.width, bbox[3]["y"] * page.rect.height),
                ).rect
                
                highlight = page.add_highlight_annot(rect)
                highlight.update()
                
                # Build tooltip entry
                tooltips.append({
                    "page": page_num + 1,
                    "x": bbox[0]["x"],
                    "y": bbox[0]["y"],
                    "prediction": seg.get("predicted_label", "violation"),
                    "confidence": seg.get("confidence", 0.0)
                })

        # === Step 3: Save annotated PDF ===
        annotated_path = f"/tmp/{file_id}_highlighted.pdf"
        doc.save(annotated_path)

        # === Step 4: Save tooltips JSON ===
        tooltips_path = f"/tmp/tooltips_{file_id}.json"
        with open(tooltips_path, "w") as f:
            json.dump({"tooltips": tooltips}, f, indent=2)

        # === Step 5: Upload outputs ===
        annotated_blob = f"outputs/highlighted/{file_id}_highlighted.pdf"
        tooltips_blob = f"outputs/tooltips/tooltips_{file_id}.json"
        
        upload_blob(BUCKET_NAME, annotated_path, annotated_blob)
        upload_blob(BUCKET_NAME, tooltips_path, tooltips_blob)
        
        logger.info(f"[ANNOTATE] Uploaded annotated PDF and tooltips for {file_id}")
        
        return {
            "status": "success",
            "annotated_pdf_blob": annotated_blob,
            "tooltips_blob": tooltips_blob
        }
        
    except Exception as e:
        logger.error(f"[ANNOTATE] Failed annotation for {file_id}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

# === Main CLI Entrypoint ===
def main(file_id: str, violations_blob: str = None):
    if not violations_blob:
        violations_blob = f"outputs/violations/{file_id}_violations.json"
    
    try:
        result = annotate_pdf(file_id, violations_blob)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python annotate.py <file_id> <violations_blob>"}))
        sys.exit(1)
    
    file_id = sys.argv[1]
    violations_blob = sys.argv[2]
    
    try:
        result = annotate_pdf(file_id, violations_blob)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e), "status": "error"}))
        sys.exit(1)
