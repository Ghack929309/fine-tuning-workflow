import json
import os
from datetime import datetime


def load_json_data(json_path):
    """Load design data from JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)


def create_html_template(design_data, output_path):
    """Generate HTML file with positioned elements"""
    dimensions = design_data["design"]["dimensions"]
    elements = design_data["design"]["elements"]
    background = design_data["design"]["background"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Design Reconstruction</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        .design-container {{
            position: relative;
            width: {dimensions["width"]}px;
            height: {dimensions["height"]}px;
            background: {background.get("content", "#ffffff")};
            margin: 0 auto;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        
        .design-element {{
            position: absolute;
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        .shape-element {{
            background: transparent;
            border: none;
        }}
        
        .text-element {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            white-space: pre-wrap;
            line-height: 1.2;
        }}
        
        .image-element {{
            object-fit: contain;
        }}
    </style>
</head>
<body>
    <div class="design-container">
"""

    # Process all elements
    for element in elements:
        if "shapes" in element:
            for shape in element["shapes"]:
                styles = shape.get("styles", {})
                html_content += f"""
                <div class="design-element shape-element"
                     style="left: {shape["position"]["x"]}px;
                            top: {shape["position"]["y"]}px;
                            width: {shape["dimensions"]["width"]}px;
                            height: {shape["dimensions"]["height"]}px;
                            background-color: {styles.get("backgroundColor", "transparent")};
                            border: {styles.get("border", "none")};
                            border-radius: {styles.get("borderRadius", "0")}px;
                            transform: {styles.get("transform", "none")};
                            opacity: {styles.get("opacity", "1")};
                            box-shadow: {styles.get("boxShadow", "none")};">
                </div>
                """
        else:
            element_type = element.get("type", "unknown")
            styles = element.get("styles", {})
            position = element.get("position", {"x": 0, "y": 0})
            dimensions = element.get("dimensions", {"width": "auto", "height": "auto"})

            if element_type == "text":
                html_content += f"""
                <div class="design-element text-element"
                     style="left: {position["x"]}px;
                            top: {position["y"]}px;
                            width: {dimensions["width"]}px;
                            height: {dimensions["height"]}px;
                            color: {styles.get("color", "#000000")};
                            background-color: {styles.get("backgroundColor", "transparent")};
                            font-family: {styles.get("fontFamily", "Arial, sans-serif")};
                            font-size: {styles.get("fontSize", "14")}px;
                            font-weight: {styles.get("fontWeight", "normal")};
                            letter-spacing: {styles.get("letterSpacing", "normal")};
                            text-align: {styles.get("textAlign", "left")};
                            transform: {styles.get("transform", "none")};
                            opacity: {styles.get("opacity", "1")};
                            z-index: {styles.get("zIndex", "auto")};">
                    {element.get("content", "")}
                </div>
                """
            elif element_type == "image":
                html_content += f"""
                <img class="design-element image-element"
                     src="data:image/png;base64,{element.get("content", "")}"
                     style="left: {position["x"]}px;
                            top: {position["y"]}px;
                            width: {dimensions["width"]}px;
                            height: {dimensions["height"]}px;
                            opacity: {styles.get("opacity", "1")};
                            transform: {styles.get("transform", "none")};
                            object-fit: {styles.get("objectFit", "contain")};
                            z-index: {styles.get("zIndex", "auto")};">
                """

    html_content += """
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)


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
