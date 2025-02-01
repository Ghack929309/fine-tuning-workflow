import json
import os
from datetime import datetime

import cv2
import numpy as np
import pytesseract

# Configuration
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_image(image_path):
    """Load and preprocess image for analysis"""
    print("Loading and preprocessing image...")
    original = cv2.imread(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return original, thresh


def detect_background_shapes(image, num_colors=5, min_area=100):
    """Detect solid color background regions with hierarchy"""
    print("Detecting background shapes...")
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    hex_colors = [f"#{c[2]:02x}{c[1]:02x}{c[0]:02x}" for c in centers]

    shapes = []
    for i, color in enumerate(hex_colors):
        mask = (labels == i).reshape(image.shape[:2]).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for j, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < min_area:
                continue
            if hierarchy[0][j][3] != -1:
                continue  # Skip child contours

            x, y, w, h = cv2.boundingRect(cnt)
            shapes.append(
                {
                    "type": "shape",
                    "position": {"x": x, "y": y},
                    "dimensions": {"width": w, "height": h},
                    "styles": {"backgroundColor": color, "zIndex": len(shapes)},
                }
            )

    # Sort by size and position
    shapes.sort(
        key=lambda s: (
            -s["dimensions"]["width"] * s["dimensions"]["height"],
            s["position"]["y"],
        )
    )
    return shapes


def extract_text_elements(original, processed):
    """Extract text elements with original background colors"""
    print("Extracting text elements...")
    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    text_elements = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        # Get original region for color analysis
        original_roi = original[y : y + h, x : x + w]
        processed_roi = processed[y : y + h, x : x + w]

        # Color extraction
        bg_color = extract_dominant_color(original_roi)
        text = pytesseract.image_to_string(processed_roi, config="--psm 6").strip()

        if text:
            text_elements.append(
                {
                    "type": "text",
                    "content": text,
                    "position": {"x": x, "y": y},
                    "dimensions": {"width": w, "height": h},
                    "styles": {
                        "backgroundColor": bg_color,
                        "color": "#000000",
                        "fontSize": 12,
                        "zIndex": 1000 + len(text_elements),
                    },
                }
            )

    return text_elements


def extract_dominant_color(roi):
    """Improved color extraction from ROI"""
    if roi.size == 0:
        return "#000000"

    try:
        pixels = np.float32(roi.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, _, centers = cv2.kmeans(
            pixels, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        return f"#{centers[0][2]:02x}{centers[0][1]:02x}{centers[0][0]:02x}"
    except:
        return "#000000"


def save_design_data(elements, dimensions, output_dir=OUTPUT_DIR):
    """Save design data to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(output_dir, f"design_{timestamp}.json")

    design_data = {
        "design": {
            "dimensions": dimensions,
            "elements": elements,
            "metadata": {"generated_at": timestamp, "version": "1.1"},
        }
    }

    with open(output_path, "w") as f:
        json.dump(design_data, f, indent=2)

    return output_path


if __name__ == "__main__":
    # Main workflow
    image_path = os.path.join(ASSETS_DIR, "design.png")
    original, processed = load_and_preprocess_image(image_path)
    height, width = original.shape[:2]

    # Detect elements
    background_shapes = detect_background_shapes(original)
    text_elements = extract_text_elements(original, processed)

    # Combine elements (backgrounds first)
    all_elements = background_shapes + text_elements

    # Save and report
    json_path = save_design_data(all_elements, {"width": width, "height": height})
    print(f"Design data saved to: {json_path}")
