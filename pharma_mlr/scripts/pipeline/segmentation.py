# scripts/pipeline/segmentation.py
import os
import json
import logging
import re
import numpy as np
import time
from typing import List, Dict, Tuple
from pathlib import Path
from google.cloud import vision, storage
from scripts.utils.gcs_helpers import download_blob, upload_blob
from scripts.utils.segmentation_config import load_segmentation_config

# Environment variables with defaults
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlr_upload")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "/app/models")
SEGMENTATION_CONFIG_PATH = os.path.join(MODEL_BASE_PATH, "../configs/segmentation_config.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/segmentation.log')
    ]
)

logger = logging.getLogger(__name__)

class Tokenizer:
    """Enhanced tokenizer with bounding box support"""
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
    
    def wait_for_ocr_output(self, storage_client, prefix: str, max_wait_seconds: int = 60) -> bool:
        """Wait until OCR output JSON files exist"""
        start = time.time()
        while time.time() - start < max_wait_seconds:
            blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=prefix))
            if any(blob.name.endswith('.json') for blob in blobs):
                return True
            logger.info("[SEG] Waiting for OCR output...")
            time.sleep(2)
        return False
    
    def extract_tokens(self, gcs_uri: str) -> Tuple[List[Dict], str]:
        """Extract tokens with bounding boxes and full text from PDF"""
        output_prefix = f"temp_ocr/{os.path.basename(gcs_uri)}/"
        
        storage_client = storage.Client()
        blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=output_prefix))
        
        if blobs:
            logger.info(f"[SEG] OCR output already exists for {gcs_uri}, skipping OCR.")
        else:
            logger.info(f"[SEG] Running OCR for {gcs_uri}")
            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            gcs_source = vision.GcsSource(uri=gcs_uri)
            input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
            gcs_dest = vision.GcsDestination(uri=f"gs://{BUCKET_NAME}/{output_prefix}")
            output_config = vision.OutputConfig(gcs_destination=gcs_dest, batch_size=1)
            
            request = vision.AsyncAnnotateFileRequest(
                features=[feature],
                input_config=input_config,
                output_config=output_config
            )
            
            operation = self.client.async_batch_annotate_files(requests=[request])
            operation.result(timeout=300)  # Wait for request to be accepted
            
            # Wait until OCR output appears
            if not self.wait_for_ocr_output(storage_client, output_prefix):
                raise RuntimeError("OCR output not ready after timeout.")
            
            blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=output_prefix))
        
        tokens = []
        full_text = ""
        
        for blob in blobs:
            if not blob.name.endswith('.json'):
                continue
            
            content = blob.download_as_string()
            response = json.loads(content)
            
            for page in response.get('responses', []):
                page_num = page.get('pageNumber', 1)
                annotation = page.get('fullTextAnnotation', {})
                full_text += annotation.get('text', '') + '\f'
                
                for block in annotation.get('pages', [{}])[0].get('blocks', []):
                    for paragraph in block.get('paragraphs', []):
                        for word in paragraph.get('words', []):
                            word_text = ''.join([s.get('text', '') for s in word.get('symbols', [])])
                            vertices = word.get('boundingBox', {}).get('normalizedVertices', [])
                            
                            if len(vertices) == 4:
                                tokens.append({
                                    'text': word_text,
                                    'bbox': vertices,
                                    'page': page_num,
                                    'confidence': word.get('confidence', 0)
                                })
        
        return tokens, full_text.strip()

def align_segments_with_tokens(segments: List[Dict], tokens: List[Dict]) -> List[Dict]:
    """Align segments with token bounding boxes"""
    aligned_segments = []
    
    for segment in segments:
        segment_text = segment['text'].lower()
        page = segment.get('page', 1)
        page_tokens = [t for t in tokens if t['page'] == page]
        matched_tokens = []
        
        for token in page_tokens:
            if token['text'].lower() in segment_text:
                matched_tokens.append(token)
        
        if matched_tokens:
            bboxes = [t['bbox'] for t in matched_tokens]
            x_coords = [pt['x'] for box in bboxes for pt in box]
            y_coords = [pt['y'] for box in bboxes for pt in box]
            
            segment['bbox'] = [
                {'x': min(x_coords), 'y': min(y_coords)},
                {'x': max(x_coords), 'y': min(y_coords)},
                {'x': max(x_coords), 'y': max(y_coords)},
                {'x': min(x_coords), 'y': max(y_coords)}
            ]
            
            segment['tokens'] = matched_tokens
            aligned_segments.append(segment)
    
    return aligned_segments

def segment_document(text: str, config: Dict) -> List[Dict]:
    """Segment document based on configuration rules"""
    segments = []
    pages = text.split('\f')
    
    for page_num, page_text in enumerate(pages, 1):
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        
        for rule_name, rule_config in config.items():
            patterns = rule_config.get('patterns', [])
            max_sentences = rule_config.get('max_sentences', 1)
            
            for i, line in enumerate(lines):
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                    segment_text = ' '.join(lines[i:i + max_sentences])
                    segments.append({
                        'text': segment_text,
                        'page': page_num,
                        'rule': rule_name,
                        'pattern': patterns[0]
                    })
    
    return segments

def run_segmentation(file_id: str) -> Dict:
    """Complete segmentation pipeline"""
    try:
        logger.info(f"[SEG] Starting segmentation for {file_id}")
        output_blob = f"outputs/segmentation/{file_id}_segments.json"
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        if bucket.blob(output_blob).exists():
            logger.info(f"[SEG] Segmentation output already exists for {file_id}, skipping segmentation.")
            return {
                'status': 'success',
                'segments_blob': output_blob,
                'segment_count': -1
            }
        
        tokenizer = Tokenizer()
        gcs_pdf_uri = f"gs://{BUCKET_NAME}/promo_uploads/{file_id}.pdf"
        tokens, full_text = tokenizer.extract_tokens(gcs_pdf_uri)
        logger.info(f"[SEG] Extracted {len(tokens)} tokens")
        
        config = load_segmentation_config(SEGMENTATION_CONFIG_PATH)
        segments = segment_document(full_text, config)
        logger.info(f"[SEG] Identified {len(segments)} segments")
        
        aligned_segments = align_segments_with_tokens(segments, tokens)
        
        output_path = f"/tmp/{file_id}_segments.json"
        with open(output_path, 'w') as f:
            json.dump({
                'file_id': file_id,
                'segments': aligned_segments,
                'token_count': len(tokens),
                'segment_count': len(aligned_segments)
            }, f, indent=2)
        
        upload_blob(BUCKET_NAME, output_path, output_blob)
        logger.info(f"[SEG] Uploaded segments to gs://{BUCKET_NAME}/{output_blob}")
        
        return {
            'status': 'success',
            'segments_blob': output_blob,
            'segment_count': len(aligned_segments)
        }
        
    except Exception as e:
        logger.error(f"[SEG] Segmentation failed: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }

def main(file_id=None):
    if file_id is None:
        import sys
        if len(sys.argv) < 2:
            print(json.dumps({"error": "Missing file_id parameter"}))
            sys.exit(1)
        file_id = sys.argv[1]
    
    result = run_segmentation(file_id)
    print(json.dumps(result))
    return result

if __name__ == "__main__":
    main()
