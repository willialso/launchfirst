import os
import json
import time
from google.cloud import vision_v1 as vision
from scripts.utils.gcs_utils import download_blob

def run_ocr(gcs_input_uri: str, gcs_output_uri: str):
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=gcs_input_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    gcs_dest = vision.GcsDestination(uri=gcs_output_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_dest, batch_size=1)
    request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )
    op = client.async_batch_annotate_files(requests=[request])
    op.result(timeout=300)

def load_ocr_results_from_gcs(gcs_prefix: str, as_text: bool = False):
    from google.cloud import storage
    storage_client = storage.Client()
    bucket_name = "mlr_upload"
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    json_blobs = [b for b in blobs if b.name.endswith(".json")]

    all_data = []
    for b in json_blobs:
        tmp_path = f"/tmp/{os.path.basename(b.name)}"
        b.download_to_filename(tmp_path)
        with open(tmp_path, "r") as f:
            data = json.load(f)
            if "responses" in data:
                all_data.append(data)

    if as_text:
        return "\n".join(
            r.get("fullTextAnnotation", {}).get("text", "")
            for d in all_data for r in d.get("responses", [])
        )
    else:
        return all_data
