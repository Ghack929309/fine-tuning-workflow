import json
import os
from datetime import datetime
from typing import Dict, Union


class StyleGenerator:
    @staticmethod
    def get_position_styles(
        position: Dict[str, int], dimensions: Dict[str, Union[int, str]]
    ) -> Dict[str, str]:
        top = position.get("y", 0)
        left = position.get("x", 0)
        width = dimensions.get("width", "auto")
        height = dimensions.get("height", "auto")
        return {
            "position": "absolute",
            "left": f"{left}px",
            "top": f"{top}px",
            "width": f"{'auto' if not width.isdigit() else width + 'px'}",
            "height": f"{'auto' if not height.isdigit() else height + 'px'}",
        }

    @staticmethod
    def get_visual_styles(styles: Dict[str, str]) -> Dict[str, str]:
        return {
            "z-index": styles.get("z-index", 0),
            "opacity": styles.get("opacity", 1),
            "transform": styles.get("transform", "none"),
            "background-color": styles.get("background-color", "transparent"),
        }

    @staticmethod
    def get_typography_styles(styles: Dict[str, str]) -> Dict[str, str]:
        return {
            "color": styles.get("color", "transparent"),
            "font-size": f"{styles.get('font-size', 12)}px",
            "font-family": styles.get("font-family", "Arial, sans-serif"),
            "font-weight": styles.get("font-weight", "normal"),
            "font-style": styles.get("font-style", "normal"),
            "letter-spacing": styles.get("letter-spacing", "normal"),
            "text-align": styles.get("text-align", "left"),
            "line-height": styles.get("line-height", "normal"),
        }

    @staticmethod
    def get_box_styles(styles: Dict[str, str]) -> Dict[str, str]:
        return {
            "border-radius": f"{styles.get('border-radius', 0)}px",
            "box-shadow": styles.get("box-shadow", "none"),
            "object-fit": styles.get("object-fit", "contain"),
        }

    @classmethod
    def generate_style_string(
        cls,
        styles: Dict[str, str],
        position: Dict[str, int],
        dimensions: Dict[str, Union[int, str]],
    ) -> str:
        """Generate CSS style string from element properties"""
        all_styles = {
            **cls.get_position_styles(position, dimensions),
            **cls.get_visual_styles(styles),
            **cls.get_typography_styles(styles),
            **cls.get_box_styles(styles),
        }
        return "; ".join(f"{k}: {v}" for k, v in all_styles.items() if v is not None)


class HTMLTemplate:
    @staticmethod
    def get_css() -> str:
        return """
        body {
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-family: Arial, sans-serif;
        }
        .design-container {
            position: relative;
            margin: 20px auto;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .design-element {
            position: absolute;
            box-sizing: border-box;
            transition: all 0.3s ease;
        }
        .shape-element {
            border: 1px solid rgba(0,0,0,0.1);
        }
        .text-element {
            white-space: pre-wrap;
            pointer-events: none;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }
        .image-element {
            object-fit: contain;
        }
        .hidden {
            opacity: 0 !important;
            pointer-events: none;
        }
        """

    @classmethod
    def create_element(
        cls, element_type: str, content: str, style_str: str, visibility_class: str
    ) -> str:
        if element_type == "image":
            return f"""
        <img class="design-element image-element{visibility_class}"
             src="{content}"
             style="{style_str}"
             alt="Design element">"""

        return f"""
        <div class="design-element {element_type}-element{visibility_class}"
             style="{style_str}">{content if element_type == "text" else ""}</div>"""


class DesignParser:
    def __init__(self, design_data: Dict):
        self.design_data = design_data
        self.design = design_data.get("design", {})
        self.dimensions = self.design.get("dimensions", {"width": 800, "height": 600})
        self.background = self.design.get("background", {"content": "#ffffff"})
        self.elements = sorted(
            self.design.get("elements", []),
            key=lambda x: x.get("styles", {}).get("z-index", 0),
        )

    def generate_html(self) -> str:
        container_style = f"""
        width: {self.dimensions.get("width")}px; 
        height: {self.dimensions.get("height")}px; 
        background-color: {self.background.get("content", "#ffffff")};"""

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '    <meta charset="UTF-8">',
            "    <title>Design Reconstruction</title>",
            "    <style>",
            HTMLTemplate.get_css(),
            "    </style>",
            "</head>",
            "<body>",
            f'    <div class="design-container" style="{container_style}">',
        ]

        for element in self.elements:
            elem_type = element.get("type", "unknown")
            if elem_type not in ["text", "shape", "image"]:
                continue

            styles = element.get("styles", {})
            position = element.get("position", {})
            dimensions = element.get("dimensions", {})
            is_visible = element.get("isVisible", True)
            content = element.get("content", "")

            style_str = StyleGenerator.generate_style_string(
                styles, position, dimensions
            )
            visibility_class = " hidden" if not is_visible else ""

            html_parts.append(
                HTMLTemplate.create_element(
                    elem_type, content, style_str, visibility_class
                )
            )

        html_parts.extend(["    </div>", "</body>", "</html>"])

        return "\n".join(html_parts)


def create_html_template(design_data: Dict, output_path: str) -> None:
    """Generate HTML from design data with proper layering"""
    try:
        parser = DesignParser(design_data)
        html_content = parser.generate_html()

        with open(output_path, "w") as f:
            f.write(html_content)

    except Exception as e:
        import traceback

        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise Exception(f"Error generating HTML: {str(e)}")


def load_json_data(json_path: str) -> Dict:
    """Load design data from JSON file"""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"JSON file not found: {json_path}")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON format in file: {json_path}")


def main() -> None:
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
