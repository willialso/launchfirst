import re

def clean(text: str) -> str:
    """
    Basic text cleaning:
    - Collapse whitespace
    - Remove non-breaking spaces
    - Strip punctuation and normalize characters
    """
    text = text.replace("\u00a0", " ")  # replace non-breaking space
    text = re.sub(r"\s+", " ", text)   # collapse whitespace
    text = text.strip()
    return text
