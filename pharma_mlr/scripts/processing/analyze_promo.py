#!/usr/bin/env python3
"""
scripts/processing/analyze_promo.py

Modular pipeline with CSV & basic LIME, robust to missing segment fields:
 1) Download promo PDF from GCS
 2) OCR → segmentation via SciBERT chunk model
 3) Ensure each segment has bbox and page (fallback full-page)
 4) Evaluate each segment with CFR layer (binary + LIME explanation)
 5) Dump CSV of all segments (file_id, index, page, bbox_x0,...,prediction,confidence,rule,explanation)
 6) Highlight & annotate violations on PDF
 7) Upload annotated PDF, tooltip JSON, and CSV; return signed URLs
"""
import os
import json
import csv
from google.cloud import storage
import fitz  # PyMuPDF

from scripts.utils.gcs_utils import generate_signed_url, upload_blob
from scripts.utils.segmentation_pipeline import async_detect_document
from scripts.utils.ocr import load_ocr_results_from_gcs
from scripts.utils.cfr_layer import evaluate_cfr

BUCKET_NAME = os.getenv("GCS_BUCKET", "mlr_upload")
storage_client = storage.Client()

def annotate_pdf_with_tooltips(input_pdf, annotations, output_pdf):
    """
    Add rectangle and popup annotations for each violation.
    """
    doc = fitz.open(input_pdf)
    for ann in annotations:
        page = doc[ann["page"] - 1]
        x0, y0, x1, y1 = ann["bbox"]
        # highlight rectangle
        rect = fitz.Rect(x0, y0, x1, y1)
        h = page.add_rect_annot(rect)
        h.set_colors(stroke=(1, 0, 0))
        h.update()
        # tooltip icon
        icon = fitz.Rect(x0 - 5, y0 - 12, x0 + 7, y0 + 2)
        page.add_circle_annot(icon)
        # popup content: rule + confidence + LIME
        content = f"Rule: {ann['rule']}\nConfidence: {ann['confidence']:.2f}\n\n"
        for feat, weight in ann.get("explanation", []):
            content += f"{feat}: {weight:.3f}\n"
        page.add_popup_annot(icon, content)
    doc.save(output_pdf)


def analyze_promo_pipeline(file_id: str, selected_layers: list) -> dict:
    """
    Orchestrate the full pipeline and return signed URLs for PDF, JSON, CSV.
    """
    # 1) Download PDF
    gcs_in = f"promo_uploads/{file_id}.pdf"
    local_in = f"/tmp/{file_id}.pdf"
    storage_client.bucket(BUCKET_NAME).blob(gcs_in) \
        .download_to_filename(local_in)

    # 2) OCR and segmentation
    ocr_prefix = f"ocr_outputs/{file_id}"
    async_detect_document(
        f"gs://{BUCKET_NAME}/{gcs_in}",
        f"gs://{BUCKET_NAME}/{ocr_prefix}/"
    )
    full_text = load_ocr_results_from_gcs(ocr_prefix)
    from scripts.models.chunk_model import segment_with_chunk_model
    segments = segment_with_chunk_model(full_text, file_id)

    # 3) Ensure bbox/page for all segments
    pdf = fitz.open(local_in)
    page0 = pdf[0]
    default_bbox = [0, 0, page0.rect.width, page0.rect.height]
    for seg in segments:
        if 'bbox' not in seg or not isinstance(seg['bbox'], list):
            seg['bbox'] = default_bbox
        if 'page' not in seg or not isinstance(seg['page'], int):
            seg['page'] = 1

    # 4) Evaluate segments with CFR
    results = []
    annotations = []
    for idx, seg in enumerate(segments):
        res = evaluate_cfr(seg['text'])
        # record for CSV
        results.append({
            'segment_index': idx,
            'page':          seg['page'],
            'bbox':          seg['bbox'],
            'prediction':    res['prediction'],
            'confidence':    res['confidence'],
            'rule':          res['rule'],
            'explanation':   res.get('explanation', [])
        })
        # collect violations for PDF
        if res['prediction'] == 'violation':
            annotations.append({
                'page':       seg['page'],
                'bbox':       seg['bbox'],
                'rule':       res['rule'],
                'confidence': res['confidence'],
                'explanation':res.get('explanation', [])
            })

    # 5) Dump CSV to temp and upload
    csv_dir = '/tmp/outputs'
    os.makedirs(csv_dir, exist_ok=True)
    csv_local = f"{csv_dir}/cfr_results_{file_id}.csv"
    with open(csv_local, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'segment_index','page',
            'bbox_x0','bbox_y0','bbox_x1','bbox_y1',
            'prediction','confidence','rule','explanation'
        ])
        for r in results:
            x0, y0, x1, y1 = r['bbox']
            exp_str = ';'.join(f"{feat}:{weight:.3f}" for feat, weight in r['explanation'])
            writer.writerow([
                r['segment_index'], r['page'],
                x0, y0, x1, y1,
                r['prediction'], f"{r['confidence']:.6f}",
                r['rule'], exp_str
            ])
    csv_blob = f"promo_csv/{file_id}_cfr_results.csv"
    upload_blob(BUCKET_NAME, csv_local, csv_blob)
    signed_csv_url = generate_signed_url(BUCKET_NAME, csv_blob)

    # 6) Annotate PDF and upload
    annotated_pdf = f"/tmp/{file_id}_cfr_annotated.pdf"
    annotate_pdf_with_tooltips(local_in, annotations, annotated_pdf)
    pdf_blob = f"promo_highlighted/{file_id}_cfr_annotated.pdf"
    upload_blob(BUCKET_NAME, annotated_pdf, pdf_blob)

    # 7) Write tooltip JSON and upload
    tp_local = f"/tmp/{file_id}_cfr_tooltips.json"
    with open(tp_local, 'w') as f:
        json.dump(annotations, f)
    tp_blob = f"promo_tooltips/{file_id}_cfr_tooltips.json"
    upload_blob(BUCKET_NAME, tp_local, tp_blob)

    # 8) Return all signed URLs
    return {
        'signed_pdf_url':     generate_signed_url(BUCKET_NAME, pdf_blob),
        'signed_tooltip_url': generate_signed_url(BUCKET_NAME, tp_blob),
        'signed_csv_url':     signed_csv_url
    }
