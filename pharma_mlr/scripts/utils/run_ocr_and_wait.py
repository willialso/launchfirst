import time
import logging
from google.cloud import vision_v1

def run_ocr_pipeline(input_gcs_uri: str, output_gcs_prefix: str, wait_time: int = 30) -> str:
    """
    Triggers OCR on a single PDF and waits for result.
    Returns the output GCS URI containing the OCR results.
    """
    logging.info(f"OCR Trigger for {input_gcs_uri} -> Output to {output_gcs_prefix}")

    client = vision_v1.ImageAnnotatorClient()

    feature = vision_v1.Feature(type_=vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision_v1.GcsSource(uri=input_gcs_uri)
    input_config = vision_v1.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")

    gcs_destination = vision_v1.GcsDestination(uri=output_gcs_prefix)
    output_config = vision_v1.OutputConfig(gcs_destination=gcs_destination, batch_size=2)

    async_request = vision_v1.AsyncAnnotateFileRequest(
        features=[feature],
        input_config=input_config,
        output_config=output_config,
    )

    operation = client.async_batch_annotate_files(requests=[async_request])
    logging.info("Waiting for OCR operation to complete...")
    operation.result(timeout=600)

    logging.info("OCR completed.")
    return output_gcs_prefix
