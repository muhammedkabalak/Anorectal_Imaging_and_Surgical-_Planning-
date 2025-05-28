import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# Dataset sınıfı
class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.images[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (256, 256)) / 255.0
        mask = cv2.resize(mask, (256, 256)) / 255.0

        img = torch.tensor(img).unsqueeze(0).float()       # [1, H, W]
        mask = torch.tensor(mask).unsqueeze(0).float()     # [1, H, W]

        return img, mask

# Veri yolu
image_path = r"C:\Users\memir\Downloads\images"
mask_path  = r"C:\Users\memir\Downloads\masks"

dataset = MRIDataset(image_path, mask_path)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model ve Eğitim ayarları
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Eğitim döngüsü
for epoch in range(15):  
    total_loss = 0
    for img, mask in loader:
        pred = model(img)
        loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Modeli kaydet
torch.save(model.state_dict(), "model.pth")
print("✅ Model kaydedildi: model.pth")
