import os
import uuid
import logging
from google.cloud import storage
from datetime import timedelta
from fastapi import UploadFile

BUCKET_NAME = "mlr_upload"
UPLOAD_PREFIX = "promo_uploads"
storage_client = storage.Client()

logging.basicConfig(level=logging.INFO)

def upload_file_to_gcs(file_path: str, blob_path: str, retry_count=3) -> str:
    """Upload file to GCS with retry logic"""
    for attempt in range(retry_count):
        try:
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(file_path)
            logging.info(f"[PROMO] Uploaded to GCS: {blob_path}")
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=60),
                method="GET"
            )
            return url
        except Exception as e:
            if attempt < retry_count - 1:
                logging.warning(f"Upload attempt {attempt+1} failed, retrying...")
                continue
            logging.exception("🔥 Error in upload_file_to_gcs")
            raise

async def process_promo_pdf(file_path: str, file_id: str = None) -> str:
    try:
        if file_id is None:
            file_id = str(uuid.uuid4())
        blob_path = f"{UPLOAD_PREFIX}/{file_id}.pdf"
        signed_url = upload_file_to_gcs(file_path, blob_path)
        return signed_url
    except Exception as e:
        logging.exception("🔥 Error in process_promo_pdf")
        raise
