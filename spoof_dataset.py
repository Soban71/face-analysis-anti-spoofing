import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SpoofDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        if not os.path.isdir(real_dir):
            raise ValueError(f"❌ real/ folder not found: {real_dir}")
        if not os.path.isdir(fake_dir):
            raise ValueError(f"❌ fake/ folder not found: {fake_dir}")

        # label = 1 for REAL
        for f in os.listdir(real_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(real_dir, f), 1))

        # label = 0 for FAKE
        for f in os.listdir(fake_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.samples.append((os.path.join(fake_dir, f), 0))

        if len(self.samples) == 0:
            raise ValueError("❌ No images found in spoof dataset folders!")

        print(f"✅ SpoofDataset loaded: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = cv2.imread(path)
        if img is None:
            # skip corrupt image safely
            return self.__getitem__((idx + 1) % len(self.samples))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
