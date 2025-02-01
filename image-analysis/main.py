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


def load_and_preprocess_image(image_path):
    print("loading image")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary, image


preprocessed_image, original_image = load_and_preprocess_image(
    os.path.join(ASSETS_DIR, "image-1.jpg")
)


def detect_elements(image):
    print("detecting elements")
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


detected_contours = detect_elements(preprocessed_image)


def extract_dominant_color(image):
    print("extracting dominant color")
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return "#000000"  # Return black for invalid images

    try:
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        pixels = np.float32(image.reshape(-1, 3))
        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        dominant = palette[0]  # Take the first color since n_colors = 1
        return "#{:02x}{:02x}{:02x}".format(
            int(dominant[2]), int(dominant[1]), int(dominant[0])
        )
    except Exception as e:
        print(f"Error extracting color: {e}")
        return "#000000"  # Return black as fallback


def extracting_text(image, img_thresh):
    text_elements = []
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:  # Skip small contours
            continue

        # Get original image region BEFORE preprocessing
        original_roi = image[y : y + h, x : x + w]

        # Get processed ROI for OCR
        processed_roi = img_thresh[y : y + h, x : x + w]

        # Extract background color from ORIGINAL image
        bg_color = extract_dominant_color(original_roi)

        # Perform OCR on processed image
        text = pytesseract.image_to_string(processed_roi, config="--psm 6").strip()

        text_elements.append(
            {
                "content": text,
                "position": {"x": x, "y": y},
                "dimensions": {"width": w, "height": h},
                "styles": {
                    "backgroundColor": bg_color,
                    "color": "#000000",  # Default text color
                },
            }
        )

    return text_elements


texts = extracting_text(original_image, preprocessed_image)


def post_process_text(text_elements):
    print("post processing text")
    for element in text_elements:
        element["content"] = "".join(
            e for e in element["content"] if e.isalnum() or e.isspace()
        )
        element["content"] = " ".join(element["content"].split())
    return text_elements


recognized_text = post_process_text(texts)
print(f"Recognized text: {recognized_text}")


def detect_and_group_shapes(contours, image):
    print("detecting and grouping shapes")
    grouped_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        shape_color = extract_dominant_color(roi)
        shape = {
            "position": {"x": int(x), "y": int(y)},
            "dimensions": {"width": int(w), "height": int(h)},
            "backgroundColor": shape_color,
        }
        grouped = False

        for group in grouped_shapes:
            if (shape["position"]["x"] - group["position"]["x"]) ** 2 + (
                shape["position"]["y"] - group["position"]["y"]
            ) ** 2 < 100:
                group["shapes"].append(shape)
                grouped = True
                break

        if not grouped:
            grouped_shapes.append({"shapes": [shape], "position": shape["position"]})

    return grouped_shapes


detected_shapes = detect_and_group_shapes(
    contours=detected_contours, image=original_image
)


def generate_svg_for_shapes(grouped_shapes):
    print("generating svg")
    dwg = svgwrite.Drawing("design.svg", profile="tiny")
    for group in grouped_shapes:
        for shape in group["shapes"]:
            x, y, w, h = (
                shape["position"]["x"],
                shape["position"]["y"],
                shape["dimensions"]["width"],
                shape["dimensions"]["height"],
            )
            background_color = shape.get("backgroundColor", "#000000")
            dwg.add(
                dwg.rect(
                    insert=(x, y),
                    size=(w, h),
                    fill=background_color,
                    stroke="none",
                )
            )
    return dwg.tostring()


svg_content = generate_svg_for_shapes(detected_shapes)


def detect_and_convert_images(image, contours):
    print("detecting and converting images")
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
                "styles": {
                    "width": w,
                    "height": h,
                    "backgroundColor": extract_dominant_color(roi),
                },
            }
        )
    return image_elements


detected_images = detect_and_convert_images(original_image, detected_contours)


def get_image_dimensions(image):
    print("getting image dimensions")
    height, width, _ = image.shape
    return width, height


image_width, image_height = get_image_dimensions(original_image)


def generate_json_representation(
    text_elements, shape_elements, image_elements, width, height
):
    print("generating json representation")
    design_json = {
        "id": "design_1",
        "text": "Your text content here",
        "design": {
            "id": "design_1",
            "dimensions": {"width": width, "height": height},
            "background": {
                "type": "color",
                "content": extract_dominant_color(original_image),
                "styles": {
                    "width": width,
                    "height": height,
                    "backgroundColor": extract_dominant_color(original_image),
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
    time_taken = (end_time - start_time).total_seconds() / 60
    print(f"Time taken: {time_taken:.2f} minutes")
