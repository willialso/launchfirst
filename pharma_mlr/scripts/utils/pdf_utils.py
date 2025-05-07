import fitz  # PyMuPDF
import json
import logging


def annotate_pdf_with_tooltips(input_pdf_path, output_pdf_path, segments, tooltip_json_path):
    doc = fitz.open(input_pdf_path)
    tooltip_entries = []

    for segment in segments:
        text = segment["text"]
        page_num = segment.get("page", 1) - 1  # 0-based
        page = doc.load_page(page_num)

        # Locate text
        text_instances = page.search_for(text, hit_max=1)
        if not text_instances:
            logging.warning(f"Text not found for tooltip annotation: {text[:30]}...")
            continue

        rect = text_instances[0]

        # Highlight
        highlight = page.add_highlight_annot(rect)
        highlight.set_colors(stroke=(1, 1, 0), fill=(1, 1, 0))
        highlight.update()

        # Tooltip data
        tooltip_entries.append({
            "page": page_num + 1,
            "x": rect.x0,
            "y": rect.y0,
            "label": segment.get("label", "N/A"),
            "prediction": segment.get("prediction", "violation"),
            "confidence": segment.get("confidence", None),
            "uncertainty": segment.get("uncertainty", None),
            "explanation": segment.get("explanation", [])
        })

    doc.save(output_pdf_path)
    doc.close()

    with open(tooltip_json_path, "w") as f:
        json.dump(tooltip_entries, f, indent=2)

    logging.info(f"Annotated PDF saved to {output_pdf_path}")
    logging.info(f"Tooltip JSON saved to {tooltip_json_path}")
