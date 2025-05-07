# scripts/processing/pi_processor.py
import os
import re
import json
import time
import logging
import spacy
from google.cloud import vision_v1 as vision
from google.cloud import storage

BUCKET_NAME = "mlr_upload"
EXPECTED_HEADERS = [
    "BOXED WARNING", "INDICATIONS AND USAGE", "DOSAGE AND ADMINISTRATION",
    "CONTRAINDICATIONS", "WARNINGS AND PRECAUTIONS", "ADVERSE REACTIONS", "CLINICAL STUDIES"
]

async def process_pi_pdf(file_path, file_id):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"pi/{file_id}.pdf")
    blob.upload_from_filename(file_path)
    gcs_input_uri = f"gs://{BUCKET_NAME}/pi/{file_id}.pdf"
    gcs_output_uri = f"gs://{BUCKET_NAME}/pi_output/{file_id}/"

    # Start OCR
    await async_ocr_pdf(gcs_input_uri, gcs_output_uri)
    time.sleep(30)

    # Download results
    local_txt = f"cleaned_{file_id}.txt"
    full_text = await download_ocr_results(gcs_output_uri, local_txt)
    if not full_text:
        logging.error("No text extracted")
        return

    full_text = re.sub(r'\s+', ' ', full_text).strip()
    sections = segment_pi_text(full_text)
    metadata = extract_metadata_pi(full_text)
    output = map_pi_sections_to_schema(sections, full_text, metadata)

    output_json = json.dumps(output, indent=2)
    out_blob = bucket.blob(f"pi_json/{file_id}.json")
    out_blob.upload_from_string(output_json, content_type="application/json")
    logging.info("[PI] JSON uploaded to bucket.")

async def async_ocr_pdf(gcs_input_uri, gcs_output_uri):
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    request = vision.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=vision.InputConfig(
            gcs_source=vision.GcsSource(uri=gcs_input_uri),
            mime_type="application/pdf",
        ),
        output_config=vision.OutputConfig(
            gcs_destination=vision.GcsDestination(uri=gcs_output_uri),
            batch_size=2,
        )
    )
    operation = client.async_batch_annotate_files(requests=[request])
    logging.info("[PI] Waiting for OCR operation to complete...")
    operation.result(timeout=420)

async def download_ocr_results(gcs_output_uri, local_output_file):
    storage_client = storage.Client()
    bucket_name, prefix = gcs_output_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    full_text = ""
    for blob in blobs:
        if blob.name.endswith(".json"):
            content = blob.download_as_text()
            result = json.loads(content)
            for resp in result.get("responses", []):
                full_text += resp.get("fullTextAnnotation", {}).get("text", "") + "\n"
    with open(local_output_file, "w") as f:
        f.write(full_text)
    return full_text

def segment_pi_text(full_text):
    text = re.sub(r"\[.*?\]", "", full_text)
    headers_pattern = "(" + "|".join([re.escape(h) for h in EXPECTED_HEADERS]) + ")"
    matches = list(re.finditer(headers_pattern, text, flags=re.IGNORECASE))
    sections = {}
    nlp = spacy.load("en_core_web_sm")
    for idx, match in enumerate(matches):
        header = match.group(0).upper()
        start = match.end()
        end = matches[idx+1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        doc = nlp(content)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) >= 5]
        sections[header] = sentences[0] if sentences else "N/A"
    for h in EXPECTED_HEADERS:
        if h not in sections:
            sections[h] = "N/A"
    return sections

def extract_metadata_pi(full_text):
    metadata = {
        "document_id": None,
        "drug_name": None,
        "doc_level_label": "PI_upload",
        "source_type": "upload"
    }
    match = re.search(r"^\s*([A-Z]+)(?=[™®])", full_text)
    if match:
        metadata["drug_name"] = match.group(1).strip()
    else:
        match = re.search(r"^\s*(?!HIGHLIGHTS\b)([A-Z]+)", full_text)
        metadata["drug_name"] = match.group(1).strip() if match else "UNKNOWN_DRUG"
    metadata["document_id"] = f"{metadata['drug_name']}_PI_upload_001"
    return metadata

def map_pi_sections_to_schema(sections, full_text, metadata):
    def section_obj(header):
        text = sections.get(header, "N/A")
        return {
            "section_type": header,
            "text": text,
            "data_source": f"[PI] - {header.title()}",
            "data_completeness": "complete" if text != "N/A" else "N/A"
        }
    return {
        "document_id": metadata["document_id"],
        "document_type": "PI_upload",
        "drug_name": metadata["drug_name"],
        "full_text": full_text,
        "sections": [section_obj(h) for h in EXPECTED_HEADERS]
    }
