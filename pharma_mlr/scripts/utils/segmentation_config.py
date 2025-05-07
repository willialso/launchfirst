import json
import os

def load_segmentation_config(config_path: str = "/app/configs/segmentation_config.json"):
    with open(config_path, "r") as f:
        return json.load(f)
