import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/")


def load_json_data(file_name: str):
    json_path = os.path.join(OUTPUT_DIR, file_name)
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded design from {json_path}")
    # Extract design information
    design = data["design"]
    elements = design["elements"]
    background = design["background"]

    # Start generating HTML
    html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
    html += '  <meta charset="UTF-8">\n'
    html += '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    html += "  <title>Design to HTML</title>\n"
    html += "  <style>\n"
    html += "    body {\n"
    html += "      margin: 0;\n"
    html += "      padding: 0;\n"
    html += "      width: 100%;\n"
    html += "      height: 100%;\n"
    html += "      display: flex;\n"
    html += "      justify-content: center;\n"
    html += "      align-items: center;\n"
    html += "      background-color: #fff;\n"
    html += "    }\n"
    html += "    .design-container {\n"
    html += "      width: {}px;\n".format(design["dimensions"]["width"])
    html += "      height: {}px;\n".format(design["dimensions"]["height"])
    html += "      background-color: {};\n".format(background["content"])
    html += "      position: relative;\n"
    html += "    }\n"
    html += "    .element {\n"
    html += "      position: absolute;\n"
    html += "    }\n"
    html += "  </style>\n"
    html += "</head>\n<body>\n"
    html += '  <div class="design-container">\n'

    # Generate HTML elements based on the design elements
    for element in elements:
        if "shapes" in element:  # This is a group of shapes
            for shape in element["shapes"]:
                html += '    <div class="element" style="'
                html += "top: {}px; left: {}px; width: {}px; height: {}px;".format(
                    shape["position"]["y"],
                    shape["position"]["x"],
                    shape["dimensions"]["width"],
                    shape["dimensions"]["height"],
                )
                html += '"></div>\n'  # Shapes are just divs with position and size
        elif element.get(
            "isVisible", True
        ):  # This is a regular element (text or image)
            # Get width and height from either styles or dimensions
            width = (
                element.get("styles", {}).get("width")
                or element.get("dimensions", {}).get("width")
                or 0
            )
            height = (
                element.get("styles", {}).get("height")
                or element.get("dimensions", {}).get("height")
                or 0
            )

            html += '    <div class="element" style="'
            html += "top: {}px; left: {}px; width: {}px; height: {}px;".format(
                element["position"]["y"],
                element["position"]["x"],
                width,
                height,
            )
            if element["type"] == "text":
                html += '">\n'
                html += "      <span>{}</span>\n".format(element["content"])
            elif element["type"] == "image":
                html += 'background-image: url(data:image/png;base64,{});">\n'.format(
                    element["content"]
                )
            html += "    </div>\n"

    html += "  </div>\n"
    html += "</body>\n</html>\n"

    # Save the HTML file
    html_file_name = file_name.replace(".json", ".html")
    html_path = os.path.join(OUTPUT_DIR, html_file_name)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML file saved to {html_path}")
    return html


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    design_file_name = input("Enter the name of the JSON design file: ")
    load_json_data(design_file_name)
