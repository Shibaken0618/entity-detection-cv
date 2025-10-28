import json
import os

from PIL import Image

from visualize import visualize_outputs_on_page

train_dir = "../data/train"


if __name__ == "__main__":
    for sample_id in os.listdir(train_dir):
        sample_im_path = os.path.join(train_dir, sample_id, sample_id + ".png")
        sample_json_path = os.path.join(train_dir, sample_id, sample_id + ".json")

        im = Image.open(sample_im_path).convert("RGB")
        with open(sample_json_path, "r") as f:
            sample_json = json.load(f)

        visualize_outputs_on_page(im, sample_json)
        input("press enter to continue")
