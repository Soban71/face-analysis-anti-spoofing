import torch
import torch.nn as nn
from torchvision import models

class MultiTaskFaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone (Professional choice)
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        feature_dim = 512

        # Age estimation (Regression)
        self.age_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Gender classification (Binary)
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Anti-spoofing (Real / Fake)
        self.spoof_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        age = self.age_head(features)
        gender = self.gender_head(features)
        spoof = self.spoof_head(features)

        return age, gender, spoof
