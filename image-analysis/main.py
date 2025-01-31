import base64
import json
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pytesseract
import svgwrite

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets/")

print(f"Assets directory: {ASSETS_DIR}")


def load_and_preprocess_image(image_path: str):
    # Load image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred, image


preprocessed_image, original_image = load_and_preprocess_image(
    ASSETS_DIR + "image-1.jpg"
)


def detect_elements(image):
    # Detect edges
    edges = cv2.Canny(image, 100, 200)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


detected_contours = detect_elements(preprocessed_image)


def recognize_text(image, contours: list[Any]):
    colored_elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        text = pytesseract.image_to_string(roi)
        if text.strip():
            colored_elements.append(
                {
                    "type": "text",
                    "content": text.strip(),
                    "position": {"x": x, "y": y},
                    "styles": {"width": w, "height": h},
                }
            )
    return colored_elements


recognized_text = recognize_text(preprocessed_image, contours=detected_contours)


def detect_and_group_shapes(image, contours):
    shape_elements = []
    grouped_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        shape_elements.append(
            {
                "type": "shape",
                "content": "",
                "position": {"x": x, "y": y},
                "styles": {"width": w, "height": h},
            }
        )
    # Group shapes that are close together
    for shape in shape_elements:
        grouped = False
        for group in grouped_shapes:
            if (shape["position"]["x"] - group["position"]["x"]) ** 2 + (
                shape["position"]["y"] - group["position"]["y"]
            ) ** 2 < 100:
                group["shapes"].append(shape)
                grouped = True
                break
        if not grouped:
            grouped_shapes.append({"shapes": [shape]})
    return grouped_shapes


detected_shapes = detect_and_group_shapes(
    preprocessed_image, contours=detected_contours
)

print(f"Detected shapes: {detected_shapes}")


def generate_svg_for_shapes(grouped_shapes):
    dwg = svgwrite.Drawing("design.svg", profile="tiny")
    for group in grouped_shapes:
        for shape in group["shapes"]:
            x, y, w, h = (
                shape["position"]["x"],
                shape["position"]["y"],
                shape["styles"]["width"],
                shape["styles"]["height"],
            )
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill="none", stroke="black"))
    dwg.save()
    return dwg.tostring()


svg_content = generate_svg_for_shapes(detected_shapes)


def detect_and_convert_images(image, contours):
    image_elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        _, buffer = cv2.imencode(".png", roi)
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_elements.append(
            {
                "type": "image",
                "content": image_base64,
                "position": {"x": x, "y": y},
                "styles": {"width": w, "height": h},
            }
        )
    return image_elements


image_elements = detect_and_convert_images(original_image, detected_contours)


def extract_dominant_color(roi):
    pixels = np.float32(roi.reshape(-1, 3))
    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    # Check if palette has at least one color
    if palette.shape[0] > 0:
        dominant = palette[0]  # Take the first color
    else:
        dominant = np.array([0, 0, 0])  # Default to black if no colors found

    # Ensure dominant has 3 channels
    if dominant.size == 3:
        return "#{:02x}{:02x}{:02x}".format(
            int(dominant[2]), int(dominant[1]), int(dominant[0])
        )
    else:
        return "#000000"  # Return black if the dominant color is not valid


def add_colors_to_elements(image, elements):
    for element in elements:
        x, y, w, h = (
            element["position"]["x"],
            element["position"]["y"],
            element["styles"]["width"],
            element["styles"]["height"],
        )
        roi = image[y : y + h, x : x + w]
        element["styles"]["color"] = extract_dominant_color(roi)
    return elements


colored_text_elements = add_colors_to_elements(original_image, recognized_text)
colored_shape_elements = add_colors_to_elements(original_image, detected_shapes)
# print(f"Colored text elements: {colored_text_elements}")
# print(f"Colored shape elements: {colored_shape_elements}")


def extract_styles(image, elements):
    for element in elements:
        x, y, w, h = (
            element["position"]["x"],
            element["position"]["y"],
            element["styles"]["width"],
            element["styles"]["height"],
        )
        roi = image[y : y + h, x : x + w]
        dominant_color = np.bincount(roi.reshape(-1)).argmax()
        element["styles"]["color"] = "#{:06x}".format(dominant_color)
    return elements


def get_image_dimensions(image):
    height, width, _ = image.shape
    return width, height


image_width, image_height = get_image_dimensions(original_image)


def generate_json_representation(text_elements, shape_elements, width, height):
    design_json = {
        "id": "design_1",
        "dimensions": {"width": width, "height": height},
        "background": {
            "type": "color",
            "content": "#3904AA",
            "styles": {"width": width, "height": height, "backgroundColor": "#3904AA"},
        },
        "elements": text_elements + shape_elements,
    }
    return json.dumps(design_json, indent=2)


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    result = generate_json_representation(
        text_elements=recognized_text,
        shape_elements=detected_shapes,
        width=image_width,
        height=image_height,
    )
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_dir, f"design_{timestamp}.json")
    with open(output_file_path, "w") as f:
        f.write(result)
    print(f"Json representation saved to {output_file_path}")
