import base64

import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops


def preprocess(image):
    # Convert to LAB color space for better texture analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def find_candidate_regions(image):
    # Use adaptive thresholding to find potential regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours and filter by size
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (image.shape[0] * image.shape[1]) * 0.01  # 1% of total area
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]


def is_image_region(region, image):
    x, y, w, h = region
    roi = image[y : y + h, x : x + w]

    # Texture analysis using GLCM
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]

    # Edge density
    edges = cv2.Canny(roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Color complexity
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_std = np.std(hsv, axis=(0, 1))

    # Shape regularity
    aspect_ratio = w / h
    aspect_score = 1 - min(abs(aspect_ratio - 1), 0.5)

    # Combined scoring
    scores = {
        "texture": contrast > 50 and contrast < 200,
        "edges": edge_density > 0.1 and edge_density < 0.7,
        "color": np.mean(color_std) > 15,
        "shape": 0.5 < aspect_ratio < 2.0,
    }

    return sum(scores.values()) >= 3


def detect_and_convert_images(image):
    processed = preprocess(image)
    candidates = find_candidate_regions(processed)
    images = []

    for region in candidates:
        if is_image_region(region, processed):
            x, y, w, h = region
            roi = image[y : y + h, x : x + w]

            # Convert to base64
            _, buffer = cv2.imencode(".png", roi)
            base64_img = base64.b64encode(buffer).decode("utf-8")
            cv2.imshow("roi", roi)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            images.append(
                {
                    "type": "image",
                    "position": {"x": x, "y": y},
                    "styles": {"width": w, "height": h},
                    "content": f"data:image/png;base64,{base64_img}",
                }
            )

    return images
