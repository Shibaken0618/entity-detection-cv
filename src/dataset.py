import json
import os
from typing import Dict, Literal, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data.dataset import Dataset


class DetectionDataset(Dataset):
    # added transforms for data augmentation for robustness (if necessary)
    def __init__(self, base_dir: str, split: Literal["train", "val"], transforms=None):
        self.base_dir = base_dir
        self.sample_ids = [i for i in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, i))]
        self.split = split
        self.transforms = transforms
        # classes = ["TitleBlock", "Note", "View", "background"]
        classes = ["TitleBlock", "Note", "View"]
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.label2idx = {key: val + 1 for key, val in self.label2idx.items()}  # start from 1
        self.idx2label = {key + 1: val for key, val in self.idx2label.items()}  # start from 1
        self.idx2label[0] = "background"  # background class

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, Dict[Literal["boxes", "labels"], torch.Tensor]]:

        sample_id = self.sample_ids[index]
        sample_im_path = os.path.join(
            self.base_dir, self.sample_ids[index], self.sample_ids[index] + ".png"
        )
        sample_json_path = os.path.join(
            self.base_dir, self.sample_ids[index], self.sample_ids[index] + ".json"
        )

        im = Image.open(sample_im_path).convert("RGB")
        with open(sample_json_path, "r") as f:
            sample_json = json.load(f)

        if self.transforms:
            im = self.transforms(im)
        else:
            im = T.ToTensor()(im)

        targets = {}
        if len(sample_json) > 0:
            targets["boxes"] = torch.as_tensor(
                [entity["bbox"] for entity in sample_json], dtype=torch.float32
            )
            targets["labels"] = torch.as_tensor(
                [self.label2idx[entity["class"]] for entity in sample_json], dtype=torch.int64
            )
        else:
            targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            targets["labels"] = torch.zeros(0, dtype=torch.int64)

        return im, targets

    # added image transform for data augmentation (increase model robustness)
    @staticmethod
    def get_transforms(train=True):
        """ Get image transforms for training or validation """
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            # slight change in brightness, contrast, saturation, hue, occasionally grayscale images
            transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            T.RandomGrayscale(p=0.1)
        return T.Compose(transforms)



def collate_fn(batch):
    # batch is a list of tuples (image, target), helps with batching variable size targets
    images, targets = zip(*batch)
    return list(images), list(targets)
