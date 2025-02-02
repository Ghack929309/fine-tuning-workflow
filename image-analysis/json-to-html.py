import json
import os
from datetime import datetime


def load_json_data(json_path):
    """Load design data from JSON file"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise Exception(f"JSON file not found: {json_path}")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON format in file: {json_path}")


def get_style_string(styles, position, dimensions):
    """Generate CSS style string from element properties"""
    style_props = {
        "left": f"{position.get('x', 0)}px",
        "top": f"{position.get('y', 0)}px",
        "width": f"{dimensions.get('width', 'auto')}px",
        "height": f"{dimensions.get('height', 'auto')}px",
        "z-index": styles.get("zIndex", 0),
        "opacity": styles.get("opacity", 1),
        "transform": styles.get("transform", "none"),
        "background-color": styles.get("backgroundColor", "transparent"),
        "color": styles.get("color", "#000000"),
        "font-size": f"{styles.get('fontSize', 12)}px",
        "font-family": styles.get("fontFamily", "Arial, sans-serif"),
        "font-weight": styles.get("fontWeight", "normal"),
        "letter-spacing": styles.get("letterSpacing", "normal"),
        "text-align": styles.get("textAlign", "left"),
        "border-radius": f"{styles.get('borderRadius', 0)}px",
        "box-shadow": styles.get("boxShadow", "none"),
        "object-fit": styles.get("objectFit", "contain"),
    }
    return "; ".join(f"{k}: {v}" for k, v in style_props.items() if v is not None)


def create_html_template(design_data, output_path):
    """Generate HTML from design data with proper layering"""
    try:
        design = design_data.get("design", {})
        dimensions = design.get("dimensions", {"width": 800, "height": 600})
        background = design.get("background", {"content": "#ffffff"})

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
            width: {dimensions.get("width")}px;
            height: {dimensions.get("height")}px;
            margin: 20px auto;
            background-color: {background.get("content", "#ffffff")};
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .design-element {{
            position: absolute;
            box-sizing: border-box;
            transition: all 0.3s ease;
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
            opacity: 0 !important;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div class="design-container">
"""
        # Sort elements by z-index
        elements = sorted(
            design.get("elements", []),
            key=lambda x: x.get("styles", {}).get("zIndex", 0),
        )

        # Process each element
        for element in elements:
            elem_type = element.get("type", "unknown")

            if elem_type == "text":
                styles = element.get("styles", {})
                position = element.get("position", {})
                dimensions = element.get("dimensions", {})
                is_visible = element.get("isVisible", True)
                content = element.get("content", "")

                style_str = get_style_string(styles, position, dimensions)
                visibility_class = " hidden" if not is_visible else ""

                html += f"""
        <div class="design-element text-element{visibility_class}"
             style="{style_str}">{content}</div>"""

            elif elem_type == "shape":
                styles = element.get("styles", {})
                position = element.get("position", {})
                dimensions = element.get("dimensions", {})
                is_visible = element.get("isVisible", True)

                style_str = get_style_string(styles, position, dimensions)
                visibility_class = " hidden" if not is_visible else ""

                html += f"""
        <div class="design-element shape-element{visibility_class}"
             style="{style_str}"></div>"""

            elif elem_type == "image":
                styles = element.get("styles", {})
                position = element.get("position", {})
                dimensions = element.get("dimensions", {})
                is_visible = element.get("isVisible", True)
                content = element.get("content", "")

                style_str = get_style_string(styles, position, dimensions)
                visibility_class = " hidden" if not is_visible else ""

                html += f"""
        <img class="design-element image-element{visibility_class}"
             src="data:image/png;base64,{content}"
             style="{style_str}"
             alt="Design element">"""

        html += """
    </div>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

    except Exception as e:
        import traceback

        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Error generating HTML: {str(e)}")


def main():
    """Main function to handle HTML generation"""
    try:
        # Set up output directory
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Get JSON file
        json_filename = input("Enter design JSON filename (without .json): ").strip()
        json_path = os.path.join(output_dir, f"{json_filename}.json")

        # Generate HTML path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        html_path = os.path.join(output_dir, f"design_{timestamp}.html")

        # Process
        print(f"Loading JSON from: {json_path}")
        design_data = load_json_data(json_path)

        print("Generating HTML...")
        create_html_template(design_data, html_path)

        print(f"HTML output created at: {html_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
