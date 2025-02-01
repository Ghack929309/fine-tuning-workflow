import base64
import json
import os
from datetime import datetime

import cv2
import numpy as np
import pytesseract
import svgwrite

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets/")

print("Flow started")
start_time = datetime.now()
print(f"Assets directory: {ASSETS_DIR}")


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary, image


preprocessed_image, original_image = load_and_preprocess_image(
    ASSETS_DIR + "image-1.jpg"
)


def detect_elements(image):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


detected_contours = detect_elements(preprocessed_image)


def recognize_text(image, contours):
    text_elements = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        text = pytesseract.image_to_string(roi, config="--psm 6")
        if text.strip():
            cleaned_text = "".join(e for e in text if e.isalnum() or e.isspace())
            text_elements.append(
                {
                    "type": "text",
                    "content": cleaned_text,
                    "isVisible": True,
                    "position": {"x": x, "y": y},
                    "styles": {"width": w, "height": h},
                }
            )
    return text_elements


texts = recognize_text(preprocessed_image, contours=detected_contours)


def post_process_text(text_elements):
    for element in text_elements:
        # Remove non-printable characters
        element["content"] = "".join(
            e for e in element["content"] if e.isalnum() or e.isspace()
        )
        # Normalize text (e.g., remove extra spaces)
        element["content"] = " ".join(element["content"].split())
    return text_elements


recognized_text = post_process_text(texts)
print(f"Recognized text: {recognized_text}")


def detect_and_group_shapes(contours):
    grouped_shapes = []

    for contour in contours:
        # Get bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)
        shape = {
            "position": {"x": int(x), "y": int(y)},
            "dimensions": {"width": int(w), "height": int(h)},
        }
        grouped = False

        for group in grouped_shapes:
            if "position" in shape and "position" in group:
                if (shape["position"]["x"] - group["position"]["x"]) ** 2 + (
                    shape["position"]["y"] - group["position"]["y"]
                ) ** 2 < 100:
                    group["shapes"].append(shape)
                    grouped = True
                    break

        if not grouped:
            grouped_shapes.append({"shapes": [shape], "position": shape["position"]})

    return grouped_shapes


detected_shapes = detect_and_group_shapes(contours=detected_contours)

# print(f"Detected shapes: {detected_shapes}")


def generate_svg_for_shapes(grouped_shapes):
    dwg = svgwrite.Drawing("design.svg", profile="tiny")
    for group in grouped_shapes:
        for shape in group["shapes"]:
            x, y, w, h = (
                shape["position"]["x"],
                shape["position"]["y"],
                shape["dimensions"]["width"],
                shape["dimensions"]["height"],
            )
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill="none", stroke="black"))
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
                "isVisible": True,
                "position": {"x": x, "y": y},
                "dimensions": {"width": w, "height": h},
            }
        )
    return image_elements


detected_images = detect_and_convert_images(original_image, detected_contours)


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


# def add_colors_to_elements(image, elements):
#     # Handle both grouped shapes and individual elements
#     for element in elements:
#         if "shapes" in element:  # This is a group of shapes
#             for shape in element["shapes"]:
#                 x, y, w, h = (
#                     shape["position"]["x"],
#                     shape["position"]["y"],
#                     shape["dimensions"]["width"],
#                     shape["dimensions"]["height"],
#                 )
#                 roi = image[y : y + h, x : x + w]
#                 if "styles" not in shape:
#                     shape["styles"] = {}
#                 shape["styles"]["color"] = extract_dominant_color(roi)
#         else:  # This is an individual element
#             x, y, w, h = (
#                 element["position"]["x"],
#                 element["position"]["y"],
#                 element["dimensions"]["width"],
#                 element["dimensions"]["height"],
#             )
#             roi = image[y : y + h, x : x + w]
#             if "styles" not in element:
#                 element["styles"] = {}
#             element["styles"]["color"] = extract_dominant_color(roi)
#     return elements


# colored_text_elements = add_colors_to_elements(original_image, recognized_text)
# colored_shape_elements = add_colors_to_elements(original_image, detected_shapes)
# print(f"Colored text elements: {colored_text_elements}")
# print(f"Colored shape elements: {colored_shape_elements}")


def get_image_dimensions(image):
    height, width, _ = image.shape
    return width, height


image_width, image_height = get_image_dimensions(original_image)


def generate_json_representation(
    text_elements, shape_elements, image_elements, width, height
):
    design_json = {
        "id": "design_1",
        "text": "Your text content here",
        "design": {
            "id": "design_1",
            "dimensions": {"width": width, "height": height},
            "background": {
                "type": "color",
                "content": "#3904AA",
                "styles": {
                    "width": width,
                    "height": height,
                    "backgroundColor": "#3904AA",
                },
            },
            "elements": text_elements + shape_elements + image_elements,
        },
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
        image_elements=detected_images,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_dir, f"design_{timestamp}.json")
    with open(output_file_path, "w") as f:
        f.write(result)
    print(f"Json representation saved to {output_file_path}")
    end_time = datetime.now()
    time_taken = (end_time - start_time) / 60
    print(f"Time taken: {time_taken.total_seconds():.2f} minutes")
