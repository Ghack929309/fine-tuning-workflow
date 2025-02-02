import base64
import os
from datetime import datetime

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# Load pre-trained models
yolo_model = YOLO("yolov5su.pt")
reader = easyocr.Reader(["en"])  # You can add more languages if needed


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return img, thresh


def detect_images(image):
    results = yolo_model(image)
    images = []
    mask = np.zeros_like(image)

    # Process each detection from the first result
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # Get coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])  # Get confidence score
            cls = int(box.cls[0])  # Get class ID

            # Extract the region
            img_region = image[y1:y2, x1:x2]

            # Only process if the region is valid and confidence is high enough
            if img_region.size > 0 and conf > 0.5:
                # Convert to RGB for better visualization
                img_region_rgb = cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB)
                _, img_encoded = cv2.imencode(".png", img_region_rgb)
                img_base64 = base64.b64encode(img_encoded).decode("utf-8")

                # Calculate aspect ratio
                aspect_ratio = (x2 - x1) / (y2 - y1)

                # Get dominant colors for the image region
                img_rgb = cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB)
                pixels = np.float32(img_rgb.reshape(-1, 3))
                n_colors = 3
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    200,
                    0.1,
                )
                flags = cv2.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv2.kmeans(
                    pixels, n_colors, None, criteria, 10, flags
                )
                _, counts = np.unique(labels, return_counts=True)
                colors = [
                    "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
                    for c in palette
                ]

                images.append(
                    {
                        "type": "image",
                        "content": img_base64,
                        "position": {"x": x1, "y": y1},
                        "styles": {
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "aspectRatio": round(aspect_ratio, 2),
                            "objectFit": "cover",
                            "opacity": 1.0,
                        },
                        "metadata": {
                            "confidence": round(conf, 3),
                            "class_id": cls,
                            "class_name": results[0].names[cls],
                            "dominant_colors": colors,
                        },
                    }
                )
                mask[y1:y2, x1:x2] = 255

    return images, cv2.bitwise_and(image, mask)


def detect_text(image, thresh):
    result = reader.readtext(thresh)
    texts = []
    mask = np.zeros_like(image)

    def get_text_color(img_region):
        avg_color = np.mean(img_region, axis=(0, 1))
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
        )
        return hex_color

    def get_font_properties(img_region, text, height):
        font_size = int(height * 0.7)
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)
        pixel_density = np.mean(gray)
        font_weight = "bold" if pixel_density > 127 else "normal"
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)
        font_style = (
            "italic"
            if lines is not None
            and any(abs(line[0][1] - np.pi / 4) < 0.3 for line in lines)
            else "normal"
        )
        width = img_region.shape[1]
        text_width = len(text) * (font_size * 0.6)
        margin = width - text_width
        if margin < 10:
            text_align = "justify"
        elif margin > width * 0.3:
            text_align = "center"
        else:
            text_align = "left"
        line_height = round(height / font_size, 1)
        return {
            "fontSize": f"{font_size}px",
            "fontWeight": font_weight,
            "fontStyle": font_style,
            "textAlign": text_align,
            "lineHeight": str(line_height),
        }

    for bbox, text, confidence in result:
        (tl, tr, br, bl) = bbox
        x1, y1 = int(tl[0]), int(tl[1])
        x2, y2 = int(br[0]), int(br[1])
        width = x2 - x1
        height = y2 - y1

        text_region = image[y1:y2, x1:x2]
        text_color = get_text_color(text_region)
        font_properties = get_font_properties(text_region, text, height)

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
        mask[y1:y2, x1:x2] = 255
    return texts, cv2.bitwise_and(image, mask)


def detect_shapes(image, thresh):
    # Create a mask for detected shapes
    mask = np.zeros_like(image)
    shapes = []

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Filter out very small contours
        if area < 100:  # Adjust this threshold as needed
            continue

        # Approximate the shape
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Determine shape type based on number of vertices
        if num_vertices == 3:
            shape_type = "triangle"
        elif num_vertices == 4:
            # Check if it's a square or rectangle
            aspect_ratio = float(w) / h
            shape_type = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif num_vertices == 5:
            shape_type = "pentagon"
        elif num_vertices == 6:
            shape_type = "hexagon"
        elif num_vertices > 10:
            shape_type = "circle"
        else:
            shape_type = "polygon"

        # Get the shape color
        mask_temp = np.zeros_like(image)
        cv2.drawContours(mask_temp, [contour], -1, (255, 255, 255), -1)
        mean_color = cv2.mean(image, mask=mask_temp[:, :, 0])
        color = "#{:02x}{:02x}{:02x}".format(
            int(mean_color[2]), int(mean_color[1]), int(mean_color[0])
        )

        # Add shape to mask
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        shapes.append(
            {
                "type": "shape",
                "shapeType": shape_type,
                "position": {"x": int(x), "y": int(y)},
                "styles": {
                    "width": int(w),
                    "height": int(h),
                    "fill": color,
                    "stroke": color,
                    "strokeWidth": 1,
                },
            }
        )

    # Return both the shapes and the image with shapes masked out
    return shapes, cv2.bitwise_and(image, cv2.bitwise_not(mask))


def detect_background(image):
    # Convert image to RGB format for better color analysis
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to be a list of pixels
    pixels = img_rgb.reshape(-1, 3)

    # Use K-means clustering to find the dominant colors
    # Convert to float32 for k-means
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, 5, None, criteria, 10, flags)

    # Convert counts
    _, counts = np.unique(labels, return_counts=True)

    # Get the most dominant color (background is usually the most common color)
    dominant_color = palette[np.argmax(counts)]

    # Convert RGB to hex color
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])
    )

    return hex_color


def generate_json(image_path):
    # Read and process the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create threshold image for text and shape detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Get image dimensions
    height, width = img.shape[:2]

    # Detect text and get the mask
    texts, img_no_text = detect_text(img, thresh)

    # Detect images and get the mask
    images, img_no_images = detect_images(img)

    # Detect shapes from the remaining image
    shapes, img_no_shapes = detect_shapes(img_no_text, thresh)

    # Get background color
    background_color = detect_background(img)

    # Create the JSON structure as a dictionary
    json_dict = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "imageInfo": {
            "width": width,
            "height": height,
            "background": background_color,
        },
        "elements": texts + shapes + images,  # Combine all detected elements
    }

    # Convert all numeric types to strings and format the JSON
    def serialize(obj):
        if isinstance(obj, (int, float)):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    # Create a formatted JSON string
    json_lines = ["{"]
    json_lines.append('    "version": "1.0",')
    json_lines.append(f'    "timestamp": "{datetime.now().isoformat()}",')
    json_lines.append('    "imageInfo": {')
    json_lines.append(f'        "width": "{width}",')
    json_lines.append(f'        "height": "{height}",')
    json_lines.append(f'        "background": "{background_color}"')
    json_lines.append("    },")
    json_lines.append('    "elements": [')

    # Add elements
    all_elements = texts + shapes + images
    for i, elem in enumerate(all_elements):
        # Convert element to string representation
        elem_lines = []
        elem_lines.append("        {")
        for key, value in elem.items():
            if isinstance(value, dict):
                elem_lines.append(f'            "{key}": {{')
                for k, v in value.items():
                    elem_lines.append(f'                "{k}": "{serialize(v)}",')
                elem_lines[-1] = elem_lines[-1].rstrip(",")  # Remove last comma
                elem_lines.append(
                    "            }" + ("," if key != list(elem.keys())[-1] else "")
                )
            else:
                elem_lines.append(f'            "{key}": "{serialize(value)}",')
        elem_lines[-1] = elem_lines[-1].rstrip(",")  # Remove last comma
        elem_lines.append("        }" + ("," if i < len(all_elements) - 1 else ""))
        json_lines.extend(elem_lines)

    json_lines.append("    ]")
    json_lines.append("}")

    return "\n".join(json_lines)


def main(image_path):
    json_output = generate_json(image_path)
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_dir, f"graphic_{timestamp}.json")
    with open(output_file_path, "w") as f:
        f.write(json_output)
    print(f"Json representation saved to {output_file_path}")


if __name__ == "__main__":
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets/")
    image_path = os.path.join(ASSETS_DIR, "image-1.jpg")  # Replace with your image path
    main(image_path)
