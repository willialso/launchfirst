# scripts/utils/gcs_helpers.py

import os
from google.cloud import storage
from datetime import timedelta

# Default bucket, but functions will now accept dynamic bucket_name
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlr_upload")

storage_client = storage.Client()

def upload_blob(bucket_name: str, source_file_path: str, destination_blob_path: str):
    """Upload a file to GCS"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_filename(source_file_path)
    print(f"📤 Uploaded {source_file_path} → {bucket_name}/{destination_blob_path}")

def download_blob(bucket_name: str, source_blob_path: str, destination_file_path: str):
    """Download a blob from GCS to local"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_path)
    blob.download_to_filename(destination_file_path)
    print(f"📥 Downloaded {bucket_name}/{source_blob_path} → {destination_file_path}")

def generate_signed_url(blob_path: str, expiration_minutes: int = 60) -> str:
    """Generate a signed URL for a blob"""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(expiration=timedelta(minutes=expiration_minutes))
    return url
