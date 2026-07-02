import pickle
import cv2
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def preprocess_signature(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    crop = thresh[
        y:y+h,
        x:x+w
    ]

    crop = 255 - crop

    crop = cv2.resize(
        crop,
        (224, 224)
    )

    return Image.fromarray(crop)


class CustomPairsDataset(Dataset):

    def __init__(self, pair_file):

        with open(pair_file, "rb") as f:
            self.pairs = pickle.load(f)

        self.transform = transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(
            img1_path
        ).convert("RGB")

        img2 = Image.open(
            img2_path
        ).convert("RGB")

        img1 = preprocess_signature(
            img1
        )

        img2 = preprocess_signature(
            img2
        )

        img1 = self.transform(
            img1
        )

        img2 = self.transform(
            img2
        )

        return (
            img1,
            img2,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )