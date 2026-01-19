import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.multitask_model import MultiTaskFaceModel
from dataset_loader import FaceDataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load dataset
dataset = FaceDataset("./datasets/part1/part1/")
print("Total images found:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # REQUIRED for Windows
)

model = MultiTaskFaceModel().to(device)

# Loss functions
age_loss_fn = nn.MSELoss()            # Regression
gender_loss_fn = nn.CrossEntropyLoss()
spoof_loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, ages, genders, spoofs in tqdm(loader):
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        spoofs = spoofs.to(device)

        pred_age, pred_gender, pred_spoof = model(images)

        loss_age = age_loss_fn(pred_age.squeeze(), ages)
        loss_gender = gender_loss_fn(pred_gender, genders)
        loss_spoof = spoof_loss_fn(pred_spoof, spoofs)

        loss = loss_age + loss_gender + loss_spoof

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "multitask_model.pth")

print("✅ Training completed successfully")
