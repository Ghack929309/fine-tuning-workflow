import cv2
import numpy as np


def get_text_color(img_region):
    """Detect dominant text color using edge-isolated k-means clustering"""
    try:
        # Convert to LAB color space for better color perception
        lab = cv2.cvtColor(img_region, cv2.COLOR_BGR2LAB)

        # Isolate text strokes using edge detection
        edges = cv2.Canny(img_region, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.dilate(edges, kernel, iterations=1) > 0

        # Extract colors from text regions
        pixels = lab[text_mask].reshape(-1, 3)
        if pixels.size == 0:  # Fallback for low-contrast text
            pixels = lab.reshape(-1, 3)

        # Find dominant color using optimized k-means
        k = min(3, max(1, pixels.shape[0] // 100))  # Dynamic cluster count
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Get the most frequent color cluster
        unique, counts = np.unique(labels, return_counts=True)
        dominant_lab = centers[unique[np.argmax(counts)]]

        # Convert LAB to RGB hex
        dominant_color = cv2.cvtColor(np.uint8([[dominant_lab]]), cv2.COLOR_LAB2BGR)[0][
            0
        ][::-1]  # Convert BGR to RGB

        return "#{:02x}{:02x}{:02x}".format(*dominant_color)

    except Exception:
        return "#000000"  # Fallback to black
