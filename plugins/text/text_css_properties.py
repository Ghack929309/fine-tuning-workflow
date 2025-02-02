import cv2
import numpy as np


def analyze_font_characteristics(img_region, text):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)

    # Create mask for green colors (Hue 60-90°)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Use green channel for grayscale conversion if green text is dominant
    if np.mean(green_mask) > 127:
        gray = img_region[:, :, 1]  # Green channel
    else:
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)

    # Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Multi-stage analysis
    analysis = {
        "font-weight": calculate_font_weight(gray),
        "font-family": detect_font_family(gray),
        "text-decoration": detect_text_decoration(gray),
        "letter-spacing": calculate_letter_spacing(gray),
        "font-style": detect_font_style(gray),
        "text-align": detect_text_alignment(gray, text),
        "font-size": calculate_font_size(gray),
        "line-height": calculate_line_height(gray),
    }

    return analysis


def calculate_font_weight(gray):
    """Calculate font weight using combined density and stroke analysis"""
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Stroke width transform
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    stroke_width = np.mean(dist[dist > 0])

    # Pixel density analysis
    density = np.mean(thresh) / 255

    # Combined weight score (0-1000 scale)
    weight_score = (stroke_width * 40) + ((1 - density) * 60)

    # Map to CSS weights
    if weight_score < 300:
        return "300"
    if weight_score < 450:
        return "400"
    if weight_score < 600:
        return "600"
    return "700"


def detect_font_family(gray):
    """Detect font family using structural analysis"""
    # Vertical edge detection for serifs
    edges = cv2.Canny(gray, 50, 150)
    vertical_edges = cv2.filter2D(edges, -1, np.array([[1, 0, -1]] * 3))

    # Serif detection
    serif_score = np.sum(vertical_edges) / edges.size

    # Character width variation
    contours, _ = cv2.findContours(255 - gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    widths = [cv2.boundingRect(c)[2] for c in contours]
    width_variation = np.std(widths) / np.mean(widths) if widths else 0

    # Font classification
    if serif_score > 0.15:
        return (
            "'Times New Roman', serif" if width_variation > 0.3 else "'Georgia', serif"
        )
    if width_variation < 0.15:
        return "'Courier New', monospace"
    return "'Arial', sans-serif"


def detect_text_decoration(gray):
    """Detect underlines/strikethrough using horizontal line detection"""
    edges = cv2.Canny(gray, 50, 150)
    horizontal = cv2.filter2D(edges, -1, np.ones((1, 20)) / 20)

    # Check bottom region for underline
    underline_region = horizontal[-10:-1, :]
    if np.mean(underline_region) > 0.3:
        return "underline"

    # Check middle region for strikethrough
    strike_region = horizontal[gray.shape[0] // 2 - 3 : gray.shape[0] // 2 + 3, :]
    if np.mean(strike_region) > 0.25:
        return "line-through"

    return "none"


def calculate_letter_spacing(gray):
    """Calculate letter spacing using contour analysis"""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 2:
        return "normal"

    # Get sorted x-coordinates
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])

    # Calculate spacing between characters
    spaces = [
        boxes[i + 1][0] - (boxes[i][0] + boxes[i][2]) for i in range(len(boxes) - 1)
    ]
    avg_space = np.median(spaces)

    # Convert to em units (assuming 1em = average character width)
    avg_width = np.mean([b[2] for b in boxes])
    return f"{avg_space / avg_width:.2f}em"


def detect_font_style(gray):
    """Detect italic using shear analysis"""
    moments = cv2.moments(gray)
    if moments["mu02"] == 0:
        return "normal"

    # Calculate skewness
    skew = moments["mu11"] / moments["mu02"]
    return "italic" if abs(skew) > 0.3 else "normal"


def detect_text_alignment(gray, text):
    """Detect text alignment using spatial distribution"""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    boxes = [cv2.boundingRect(c) for c in contours]
    if not boxes:
        return "left"

    # Calculate spread
    leftmost = min(b[0] for b in boxes)
    rightmost = max(b[0] + b[2] for b in boxes)
    width = rightmost - leftmost

    # Alignment heuristics
    if width < gray.shape[1] * 0.7:
        return "center"
    if (boxes[-1][0] + boxes[-1][2]) < gray.shape[1] * 0.9:
        return "right"
    return "left"


def calculate_line_height(gray):
    """Estimate line height using vertical projection"""
    # Vertical projection histogram
    hist = cv2.reduce(gray, 1, cv2.REDUCE_AVG)
    hist = hist.reshape(-1)

    # Find text baseline
    peaks = np.where(hist < np.mean(hist) * 0.8)[0]
    if len(peaks) < 2:
        return "normal"

    # Calculate line height from peak distances
    distances = np.diff(peaks)
    return f"{np.median(distances) / gray.shape[0]:.2f}em"


def calculate_font_size(gray):
    """Calculate font size using character height analysis (returns px values)"""
    # Threshold image for contour detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find character contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "12px"  # Default fallback value

    # Get heights of all detected characters
    heights = []
    for c in contours:
        _, _, _, h = cv2.boundingRect(c)
        if h > 3:  # Filter out noise
            heights.append(h)

    if not heights:
        return "12px"

    # Calculate median height in pixels and round to nearest integer
    median_height = int(np.median(heights))

    return median_height
