import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.multitask_model import MultiTaskFaceModel
from spoof_dataset import SpoofDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# load base model
model = MultiTaskFaceModel().to(device)
model.load_state_dict(torch.load("multitask_model.pth", map_location=device))
model.train()

# Freeze all except spoof head
for name, param in model.named_parameters():
    param.requires_grad = False

for param in model.spoof_head.parameters():
    param.requires_grad = True

# dataset
dataset = SpoofDataset("datasets/spoof/")
print("Total spoof samples:", len(dataset))

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.spoof_head.parameters(), lr=2e-4)  # smaller LR ok

EPOCHS = 5

for epoch in range(EPOCHS):
    total = 0
    running_loss = 0

    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, _, pred_spoof = model(imgs)

        loss = loss_fn(pred_spoof, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += 1

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/total:.4f}")

torch.save(model.state_dict(), "multitask_model_spoof.pth")
print("Saved new weights: multitask_model_spoof.pth")
