from typing import Any, Dict, List

from PIL.Image import Image as ImageType
from PIL.ImageDraw import ImageDraw

from custom_types import HitBox


def visualize_outputs_on_page(image: ImageType, sample_json: List[Dict[str, Any]]):

    hit_boxes: List[HitBox] = []
    for entity in sample_json:
        hit_boxes.append(
            {
                "class_name": entity["class"],
                "x1": entity["bbox"][0],
                "y1": entity["bbox"][1],
                "x2": entity["bbox"][2],
                "y2": entity["bbox"][3],
            }
        )

    # draw the hit boxes on the image
    draw = ImageDraw(image)
    for hit_box in hit_boxes:
        if hit_box["class_name"] == "TitleBlock":
            color = "green"
        elif hit_box["class_name"] == "View":
            color = "red"
        else:
            color = "blue"

        draw.rectangle(
            (hit_box["x1"], hit_box["y1"], hit_box["x2"], hit_box["y2"]),
            outline=color,
            width=4,
        )

    image.show()
