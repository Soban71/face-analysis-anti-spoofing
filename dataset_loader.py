# dataset_loader.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if len(self.images) == 0:
            raise ValueError("❌ No images found in dataset folder!")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self.images))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        # UTKFace format: age_gender_race_xxx.jpg
        parts = img_name.split("_")
        age = float(parts[0])
        gender = int(parts[1])

        # spoof label (dataset does NOT have it → assume real face)
        spoof = 1   # 1 = real, 0 = fake

        return (
            image,
            torch.tensor(age, dtype=torch.float32),
            torch.tensor(gender, dtype=torch.long),
            torch.tensor(spoof, dtype=torch.long)
        )
