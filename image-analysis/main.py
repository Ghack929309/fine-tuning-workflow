import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from datetime import datetime

import cv2
import easyocr
import numpy as np
import pytesseract
import svgwrite
from sklearn.cluster import DBSCAN

from plugins.image.image_detection import detect_and_convert_images
from plugins.text.color_detection import get_text_color
from plugins.text.text_css_properties import analyze_font_characteristics

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets/")

reader = easyocr.Reader(["en"])

print("Flow started")
start_time = datetime.now()


def load_and_preprocess_image(image_path):
    print("loading image")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary, image


preprocessed_image, original_image = load_and_preprocess_image(
    os.path.join(ASSETS_DIR, "image-2.jpg")
)


def detect_elements(image):
    print("detecting elements")
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


detected_contours = detect_elements(preprocessed_image)


def detect_background_colors(image):
    """Detects multiple background colors with spatial positions using DBSCAN clustering"""
    if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        return []

    try:
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Downsample image to reduce memory usage
        max_dimension = 100  # Limit maximum dimension
        h, w = image.shape[:2]
        scale = min(max_dimension / w, max_dimension / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        # Create spatial-color features (x, y, r, g, b) more efficiently
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.column_stack((x_coords.ravel() / w, y_coords.ravel() / h))
        colors = image.reshape(-1, 3).astype(float) / 255.0
        features = np.hstack([coords, colors])

        # Cluster using DBSCAN with adjusted parameters for downsampled image
        dbscan = DBSCAN(
            eps=0.1, min_samples=5
        )  # Reduced min_samples due to downsampling
        clusters = dbscan.fit_predict(features)

        # Calculate cluster statistics
        unique_clusters = np.unique(clusters[clusters != -1])
        if len(unique_clusters) == 0:
            return []

        total_pixels = len(clusters[clusters != -1])
        color_info = []

        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_points = features[mask]
            count = len(cluster_points)

            # Get median color
            median_color = np.median(cluster_points[:, 2:], axis=0)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(median_color[2] * 255),
                int(median_color[1] * 255),
                int(median_color[0] * 255),
            )

            # Get position bounds
            x_min = cluster_points[:, 0].min()
            x_max = cluster_points[:, 0].max()
            y_min = cluster_points[:, 1].min()
            y_max = cluster_points[:, 1].max()

            color_info.append(
                {
                    "color": hex_color,
                    "coverage": count / total_pixels,
                    "position": {
                        "x_start": x_min,
                        "x_end": x_max,
                        "y_start": y_min,
                        "y_end": y_max,
                    },
                }
            )

        return sorted(color_info, key=lambda x: -x["coverage"])

    except Exception as e:
        print(f"Error detecting background colors: {e}")
        return []


def detect_text(image):
    result = reader.readtext(image)  # Use original image for OCR
    texts = []

    for bbox, text, confidence in result:
        (tl, tr, br, bl) = bbox
        x1, y1 = int(tl[0]), int(tl[1])
        x2, y2 = int(br[0]), int(br[1])
        width = x2 - x1
        height = y2 - y1
        print(f"Detected text: '{text}' with confidence: {(confidence * 100):.2f}%")
        text_region = image[y1:y2, x1:x2]
        text_color = get_text_color(text_region)
        font_properties = analyze_font_characteristics(text_region, text)
        texts.append(
            {
                "type": "text",
                "content": text,
                "position": {"x": x1, "y": y1},
                "styles": {
                    "width": width,
                    "height": height,
                    "color": text_color,
                    **font_properties,
                },
            }
        )

    return texts


detected_text = detect_text(original_image)
print(f"Detected text: {detected_text}")


def extracting_text(img_thresh):
    """Extract text with transparent backgrounds and detected text color"""
    text_elements = []

    # Use RETR_EXTERNAL to get outer contours and avoid detecting individual letters
    contours, hierarchy = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours from left to right, top to bottom
    contours = sorted(
        contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0])
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:  # Skip very small contours
            continue

        # Get original and processed regions
        processed_roi = img_thresh[y : y + h, x : x + w]

        # Use --psm 7 for treating the image as a single text line
        text = pytesseract.image_to_string(
            processed_roi, config="--psm 7 --oem 3"
        ).strip()

        if text:  # Only add if text was found
            text_elements.append(
                {
                    "type": "text",
                    "content": text,
                    "isVisible": True,
                    "position": {"x": x, "y": y},
                    "dimensions": {"width": w, "height": h},
                }
            )

    return text_elements


recognized_text = extracting_text(preprocessed_image)


def detect_and_group_shapes(contours, image):
    print("detecting and grouping shapes")
    grouped_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y : y + h, x : x + w]
        colors = detect_background_colors(roi)
        shape_color = colors[0]["color"] if colors else "transparent"
        shape = {
            "position": {"x": int(x), "y": int(y)},
            "dimensions": {"width": int(w), "height": int(h)},
            "background-color": shape_color,
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
    dwg = svgwrite.Drawing("svgs/design.svg", profile="tiny")
    for group in grouped_shapes:
        for shape in group["shapes"]:
            x, y, w, h = (
                shape["position"]["x"],
                shape["position"]["y"],
                shape["dimensions"]["width"],
                shape["dimensions"]["height"],
            )
            background_color = (
                shape.get("background-color", "none")
                if shape["background-color"] != "transparent"
                else "none"
            )
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

os.makedirs("svgs", exist_ok=True)
with open("svgs/design.svg", "w") as f:
    f.write(svg_content)


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
                "content": detect_background_colors(original_image)[0]["color"]
                if detect_background_colors(original_image)
                else "transparent",
                "styles": {
                    "width": width,
                    "height": height,
                    "background-color": detect_background_colors(original_image)[0][
                        "color"
                    ]
                    if detect_background_colors(original_image)
                    else "transparent",
                },
            },
            "elements": [],
        },
    }

    # Add text elements
    for text in text_elements:
        styles = text.get("styles", {})
        css_styles = {}
        if styles:
            css_styles = {
                "width": styles.get("width"),
                "height": styles.get("height"),
                "color": styles.get("color"),
                "font-size": styles.get("font-size"),
                "font-weight": styles.get("font-weight"),
                "font-style": styles.get("font-style"),
                "text-align": styles.get("text-align"),
                "line-height": styles.get("line-height"),
                "font-family": styles.get("font-family"),
                "letter-spacing": styles.get("letter-spacing"),
            }
            # Remove None values
            css_styles = {k: v for k, v in css_styles.items() if v is not None}

        design_json["design"]["elements"].append(
            {
                "type": "text",
                "content": text["content"],
                "isVisible": text.get("isVisible", True),
                "position": text["position"],
                "styles": css_styles,
            }
        )

    # Add shape elements
    for shape_group in shape_elements:
        for shape in shape_group["shapes"]:
            design_json["design"]["elements"].append(
                {
                    "type": "shape",
                    "content": "",
                    "isVisible": True,
                    "position": shape["position"],
                    "styles": {
                        "width": shape["dimensions"]["width"],
                        "height": shape["dimensions"]["height"],
                        "background-color": shape["background-color"],
                    },
                }
            )

    # Add image elements
    for image in image_elements:
        styles = {
            "width": image["styles"]["width"],
            "height": image["styles"]["height"],
        }
        if "background-color" in image["styles"]:
            styles["background-color"] = image["styles"]["background-color"]

        design_json["design"]["elements"].append(
            {
                "type": "image",
                "content": image["content"],
                "position": image["position"],
                "styles": styles,
            }
        )

    return json.dumps(design_json, indent=2)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    image_elements = detect_and_convert_images(original_image)
    result = generate_json_representation(
        text_elements=detected_text,
        shape_elements=detected_shapes,
        image_elements=image_elements,
        width=image_width,
        height=image_height,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_dir, f"design_{timestamp}.json")
    with open(output_file_path, "w") as doc:
        doc.write(result)
    print(
        f"Json representation with name: {output_file_path.split('output/')[1].split('.')[0]}"
    )
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds() / 60
    print(f"Time taken: {time_taken:.2f} minutes")


if __name__ == "__main__":
    main()
