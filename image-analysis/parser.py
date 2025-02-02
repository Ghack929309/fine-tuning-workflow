import json
import os
from typing import Dict, List, Any
from datetime import datetime


class DesignParser:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.design_data = None

    def load_json(self) -> None:
        """Load the JSON file and store the design data"""
        with open(self.json_file_path, 'r') as f:
            self.design_data = json.load(f)

    def get_dimensions(self) -> Dict[str, str]:
        """Get the dimensions of the design"""
        return {
            'width': self.design_data['imageInfo']['width'],
            'height': self.design_data['imageInfo']['height']
        }

    def get_background(self) -> str:
        """Get the background color"""
        return self.design_data['imageInfo']['background']

    def get_elements_by_type(self, element_type: str) -> List[Dict[str, Any]]:
        """Get all elements of a specific type (text, shape, or image)"""
        return [
            element for element in self.design_data['elements']
            if element['type'] == element_type
        ]

    def get_text_elements(self) -> List[Dict[str, Any]]:
        """Get all text elements"""
        return self.get_elements_by_type('text')

    def get_shape_elements(self) -> List[Dict[str, Any]]:
        """Get all shape elements"""
        return self.get_elements_by_type('shape')

    def get_image_elements(self) -> List[Dict[str, Any]]:
        """Get all image elements"""
        return self.get_elements_by_type('image')

    def analyze_design(self) -> Dict[str, Any]:
        """Analyze the design and return statistics"""
        text_elements = self.get_text_elements()
        shape_elements = self.get_shape_elements()
        image_elements = self.get_image_elements()

        return {
            'version': self.design_data['version'],
            'timestamp': self.design_data['timestamp'],
            'dimensions': self.get_dimensions(),
            'background_color': self.get_background(),
            'element_counts': {
                'text': len(text_elements),
                'shapes': len(shape_elements),
                'images': len(image_elements)
            },
            'text_content': [elem['content'] for elem in text_elements],
            'shape_types': list(set(elem['shapeType'] for elem in shape_elements))
        }


def parse_json_to_html(json_data):
    width = json_data["imageInfo"]["width"]
    height = json_data["imageInfo"]["height"]
    background_color = json_data["imageInfo"]["background"]

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Graphic Design</title>
        <style>
            .design {{
                position: relative;
                width: {width}px;
                height: {height}px;
                background-color: {background_color};
            }}
            .element {{
                position: absolute;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .shape {{
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="design">
    """

    for element in json_data["elements"]:
        element_type = element["type"]
        x = element["position"]["x"]
        y = element["position"]["y"]
        styles = element["styles"]
        width = styles["width"]
        height = styles["height"]

        if element_type == "text":
            content = element["content"]
            color = styles.get("color", "#000000")
            font_size = styles.get("fontSize", "16px")
            font_weight = styles.get("fontWeight", "normal")
            font_style = styles.get("fontStyle", "normal")
            text_align = styles.get("textAlign", "left")
            line_height = styles.get("lineHeight", "1.2")
            
            html_content += f"""
            <div class="element" style="
                top: {y}px;
                left: {x}px;
                width: {width}px;
                height: {height}px;
                color: {color};
                font-size: {font_size};
                font-weight: {font_weight};
                font-style: {font_style};
                text-align: {text_align};
                line-height: {line_height};
            ">
                {content}
            </div>
            """

        elif element_type == "shape":
            shape_type = element["shapeType"]
            fill = styles.get("fill", "#000000")
            stroke = styles.get("stroke", fill)
            stroke_width = styles.get("strokeWidth", "1")
            
            html_content += f"""
            <div class="element shape" style="
                top: {y}px;
                left: {x}px;
                width: {width}px;
                height: {height}px;
                background-color: {fill};
                border: {stroke_width}px solid {stroke};
            " title="{shape_type}">
            </div>
            """

        elif element_type == "image":
            content = element["content"]
            aspect_ratio = styles.get("aspectRatio", "auto")
            object_fit = styles.get("objectFit", "cover")
            opacity = styles.get("opacity", "1.0")
            
            html_content += f"""
            <img class="element" 
                src="data:image/png;base64,{content}"
                style="
                    top: {y}px;
                    left: {x}px;
                    width: {width}px;
                    height: {height}px;
                    aspect-ratio: {aspect_ratio};
                    object-fit: {object_fit};
                    opacity: {opacity};
                "
            >
            """

    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content


def main(json_file_path, output_html_path):
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    html_content = parse_json_to_html(json_data)

    with open(output_html_path, "w") as html_file:
        html_file.write(html_content)

    print(f"HTML file generated at {output_html_path}")


def main_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the output directory")
        return

    latest_json = max(json_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    json_path = os.path.join(output_dir, latest_json)

    parser = DesignParser(json_path)
    parser.load_json()
    analysis = parser.analyze_design()

    print("\nDesign Analysis:")
    print(f"Version: {analysis['version']}")
    print(f"Timestamp: {analysis['timestamp']}")
    print(f"Dimensions: {analysis['dimensions']['width']}x{analysis['dimensions']['height']}")
    print(f"Background Color: {analysis['background_color']}")
    print("\nElement Counts:")
    for elem_type, count in analysis['element_counts'].items():
        print(f"- {elem_type}: {count}")
    print("\nText Content:")
    for text in analysis['text_content']:
        print(f"- {text}")
    print("\nShape Types:")
    for shape_type in analysis['shape_types']:
        print(f"- {shape_type}")


if __name__ == "__main__":
    try:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if not json_files:
            print("No JSON files found in the output directory")
            exit(1)

        latest_json = max(json_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
        json_path = os.path.join(output_dir, latest_json)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        html_path = os.path.join(output_dir, f"design_{timestamp}.html")

        print(f"Loading latest JSON from: {json_path}")
        main(json_file_path=json_path, output_html_path=html_path)
        print(f"HTML output created at: {html_path}")

        print("\nGenerating analysis...")
        main_analysis()

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
