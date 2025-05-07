#!/usr/bin/env python3

"""
GCS utility functions: upload/download blobs, generate signed URLs,
and FastAPI handlers for PDF uploads.
"""
import os
import uuid
import logging
from datetime import timedelta
from fastapi import UploadFile
from google.cloud import storage

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BUCKET_NAME    = os.getenv("GCS_BUCKET", "mlr_upload")
storage_client = storage.Client()
bucket         = storage_client.bucket(BUCKET_NAME)

# ----------------------------------------------------------------------------
# Blob operations
# ----------------------------------------------------------------------------
def upload_blob(bucket_name: str, source_file_path: str, dest_blob_name: str):
    """
    Upload a local file to GCS.
    """
    blob = storage_client.bucket(bucket_name).blob(dest_blob_name)
    blob.upload_from_filename(source_file_path)
    logging.info(f"Uploaded {source_file_path} → gs://{bucket_name}/{dest_blob_name}")

def download_blob(bucket_name: str, source_blob_name: str, dest_file_path: str):
    """
    Download a blob from GCS to a local file.
    """
    blob = storage_client.bucket(bucket_name).blob(source_blob_name)
    blob.download_to_filename(dest_file_path)
    logging.info(f"Downloaded gs://{bucket_name}/{source_blob_name} → {dest_file_path}")

def download_string(bucket_name: str, blob_name: str) -> str:
    """
    Download a blob’s contents as text (e.g. for small JSON/text files).
    """
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_text()
    logging.info(f"Fetched gs://{bucket_name}/{blob_name} as text")
    return data

def generate_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 60) -> str:
    """
    Generate a V4 signed URL for downloading a blob.
    """
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    url  = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET"
    )
    logging.info(f"Generated signed URL for gs://{bucket_name}/{blob_name}")
    return url

# ----------------------------------------------------------------------------
# FastAPI handlers for PDF uploads
# ----------------------------------------------------------------------------
async def handle_upload_promo(file: UploadFile):
    """
    Save the uploaded promotional PDF to GCS and return its file_id and signed URL.
    """
    file_id = "promo_" + uuid.uuid4().hex[:8]
    dest    = f"promo_uploads/{file_id}.pdf"
    contents = await file.read()
    bucket.blob(dest).upload_from_string(contents, content_type=file.content_type)
    logging.info(f"Uploaded promo PDF → gs://{BUCKET_NAME}/{dest}")

    signed_url = generate_signed_url(BUCKET_NAME, dest)
    return file_id, signed_url

async def handle_upload_pi(file: UploadFile):
    """
    Save the uploaded PI PDF to GCS and return its file_id and signed URL.
    """
    file_id = "pi_" + uuid.uuid4().hex[:8]
    dest    = f"pi_uploads/{file_id}.pdf"
    contents = await file.read()
    bucket.blob(dest).upload_from_string(contents, content_type=file.content_type)
    logging.info(f"Uploaded PI PDF → gs://{BUCKET_NAME}/{dest}")

    signed_url = generate_signed_url(BUCKET_NAME, dest)
    return file_id, signed_url
