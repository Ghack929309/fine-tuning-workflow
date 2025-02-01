import json
import os
from datetime import datetime


def load_json_data(json_path):
    """Load design data from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


def create_html_template(design_data, output_path):
    """Generate HTML from design data with proper layering"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Design Reconstruction</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-family: Arial, sans-serif;
        }}
        .design-container {{
            position: relative;
            width: {design_data["design"]["dimensions"]["width"]}px;
            height: {design_data["design"]["dimensions"]["height"]}px;
            margin: 20px auto;
            background-color: {design_data["design"]["background"].get("content", "#ffffff")};
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .design-element {{
            position: absolute;
            box-sizing: border-box;
            transition: opacity 0.3s ease;
        }}
        .shape-element {{
            border: 1px solid rgba(0,0,0,0.1);
        }}
        .text-element {{
            white-space: pre-wrap;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            line-height: 1.2;
        }}
        .image-element {{
            object-fit: contain;
        }}
        .hidden {{
            opacity: 0;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div class="design-container">
"""

    # Process all elements
    # First, collect and sort all elements
    all_elements = []
    
    # Add shapes
    for element in design_data["design"]["elements"]:
        if "shapes" in element:
            for shape in element["shapes"]:
                shape["type"] = "shape"
                all_elements.append(shape)
        else:
            all_elements.append(element)
    
    # Sort elements by z-index
    sorted_elements = sorted(
        all_elements,
        key=lambda x: x.get("styles", {}).get("zIndex", 0)
    )

    # Process each element
    for element in sorted_elements:
        elem_type = element.get("type", "unknown")
        styles = element.get("styles", {})
        position = element.get("position", {})
        dimensions = element.get("dimensions", {})
        is_visible = element.get("isVisible", True)
        
        # Common style attributes
        style_str = f"""
            left: {position.get("x", 0)}px;
            top: {position.get("y", 0)}px;
            width: {dimensions.get("width", "auto")};
            height: {dimensions.get("height", "auto")};
            z-index: {styles.get("zIndex", 0)};
            opacity: {styles.get("opacity", 1)};
            transform: {styles.get("transform", "none")};
        """
        
        # Visibility class
        visibility_class = " hidden" if not is_visible else ""

        if elem_type == "shape":
            html += f"""
        <div class="design-element shape-element{visibility_class}"
             style="{style_str}
                    background-color: {styles.get("backgroundColor", "transparent")};
                    border-radius: {styles.get("borderRadius", "0")}px;
                    box-shadow: {styles.get("boxShadow", "none")};">
        </div>"""

        elif elem_type == "text":
            html += f"""
        <div class="design-element text-element{visibility_class}"
             style="{style_str}
                    color: {styles.get("color", "#000000")};
                    background-color: {styles.get("backgroundColor", "transparent")};
                    font-size: {styles.get("fontSize", 12)}px;
                    font-family: {styles.get("fontFamily", "Arial, sans-serif")};
                    font-weight: {styles.get("fontWeight", "normal")};
                    letter-spacing: {styles.get("letterSpacing", "normal")};
                    text-align: {styles.get("textAlign", "left")};
                    padding: 2px;">
            {element.get("content", "")}
        </div>"""

        elif elem_type == "image":
            html += f"""
        <img class="design-element image-element{visibility_class}"
             src="data:image/png;base64,{element.get("content", "")}"
             style="{style_str}
                    object-fit: {styles.get("objectFit", "contain")};"
             alt="Design element">"""

    html += """
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Example usage
    json_filename = input("Enter design JSON path: ")
    json_path = os.path.join(output_dir, f"{json_filename}.json")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    html_path = os.path.join(output_dir, f"design_{timestamp}.html")

    design_data = load_json_data(json_path)
    create_html_template(design_data, html_path)
    print(f"HTML output created at: {html_path}")
