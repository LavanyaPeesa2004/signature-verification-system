import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        # Load ResNet18
        backbone = models.resnet18(weights="DEFAULT")

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Embedding layer
        self.embedding = nn.Linear(
            512,
            128
        )

    def forward_once(self, x):

        x = self.feature_extractor(x)

        x = torch.flatten(x, 1)

        x = self.embedding(x)

        return x

    def forward(self, img1, img2):

        emb1 = self.forward_once(img1)

        emb2 = self.forward_once(img2)

        return emb1, emb2