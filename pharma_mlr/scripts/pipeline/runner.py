#!/usr/bin/env python3
"""
Pipeline runner for MLR analysis workflow
Handles segmentation -> violation detection -> annotation
"""

import os
import sys
import json
import logging
import subprocess
from typing import Dict
from datetime import datetime

# Environment variables with defaults
BUCKET_NAME = os.getenv("BUCKET_NAME", "mlr_upload")
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "/app/models")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/pipeline_runner.log')
    ]
)

logger = logging.getLogger(__name__)

# Script paths
SEGMENTATION_SCRIPT = "scripts/pipeline/segmentation.py"
VIOLATIONS_SCRIPT = "scripts/pipeline/violations.py"
ANNOTATE_SCRIPT = "scripts/pipeline/annotate.py"

class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    def __init__(self, step: str, error: str, file_id: str):
        self.step = step
        self.error = error
        self.file_id = file_id
        super().__init__(f"Pipeline failed at {step} for {file_id}: {error}")

def validate_output(output: Dict, required_fields: list) -> bool:
    """Validate script output contains required fields"""
    if not isinstance(output, dict):
        return False
    if output.get("status") != "success":
        return False
    return all(field in output for field in required_fields)

def run_script(cmd: list, step_name: str, file_id: str) -> Dict:
    """
    Execute a pipeline script with comprehensive error handling
    Args:
        cmd: Command to execute as list
        step_name: Human-readable step name
        file_id: Current file being processed
    Returns:
        Dictionary with script output
    Raises:
        PipelineError: If script fails validation or execution
    """
    logger.info(f"[{file_id}] Starting {step_name} step")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Parse and validate output
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise PipelineError(
                step_name,
                f"Invalid JSON output: {str(e)}",
                file_id
            )
        
        # Validate required fields
        required_fields = {
            "segmentation": ["segments_blob"],
            "violations": ["violations_blob"],
            "annotation": ["annotated_pdf_blob", "tooltips_blob"]
        }.get(step_name, [])
        
        if not validate_output(output, required_fields):
            raise PipelineError(
                step_name,
                f"Invalid output format, missing required fields",
                file_id
            )
        
        logger.info(f"[{file_id}] Completed {step_name} successfully")
        return output
        
    except subprocess.CalledProcessError as e:
        error_msg = f"{step_name} failed with code {e.returncode}: {e.stderr}"
        logger.error(f"[{file_id}] {error_msg}")
        raise PipelineError(step_name, error_msg, file_id)
    except Exception as e:
        error_msg = f"Unexpected error in {step_name}: {str(e)}"
        logger.error(f"[{file_id}] {error_msg}")
        raise PipelineError(step_name, error_msg, file_id)

def run_pipeline(file_id: str) -> Dict:
    """
    Execute the complete analysis pipeline
    Args:
        file_id: Unique identifier for the document
    Returns:
        Dictionary with final results or error details
    """
    pipeline_start = datetime.now()
    logger.info(f"[{file_id}] Starting pipeline execution")
    
    try:
        # --- Segmentation ---
        seg_result = run_script(
            ["python3", SEGMENTATION_SCRIPT, file_id],
            "segmentation",
            file_id
        )
        
        # --- Violation Detection ---
        viol_result = run_script(
            ["python3", VIOLATIONS_SCRIPT, file_id, seg_result["segments_blob"]],
            "violation_detection",
            file_id
        )
        
        # --- Annotation ---
        annotate_result = run_script(
            ["python3", ANNOTATE_SCRIPT, file_id, viol_result["violations_blob"]],
            "annotation",
            file_id
        )
        
        # Calculate duration
        duration = (datetime.now() - pipeline_start).total_seconds()
        
        return {
            "status": "success",
            "file_id": file_id,
            "duration_seconds": duration,
            "annotated_pdf_blob": annotate_result["annotated_pdf_blob"],
            "tooltips_blob": annotate_result["tooltips_blob"],
            "segment_count": seg_result.get("segment_count", 0),
            "violation_count": viol_result.get("violation_count", 0)
        }
        
    except PipelineError as e:
        duration = (datetime.now() - pipeline_start).total_seconds()
        logger.error(f"[{file_id}] Pipeline failed after {duration:.2f}s")
        return {
            "status": "error",
            "file_id": file_id,
            "step": e.step,
            "error": e.error,
            "duration_seconds": duration
        }
        
    except Exception as e:
        duration = (datetime.now() - pipeline_start).total_seconds()
        logger.error(f"[{file_id}] Unexpected pipeline failure after {duration:.2f}s")
        return {
            "status": "error",
            "file_id": file_id,
            "error": str(e),
            "duration_seconds": duration
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({
            "status": "error",
            "error": "Usage: runner.py <file_id>"
        }))
        sys.exit(1)
    
    result = run_pipeline(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "success" else 1)
