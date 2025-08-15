"""
resnet18_fine_tune_flow.py

This script implements transfer learning with fine-tuning applied only to the final part of the architecture:
- Uses a ResNet-18 pretrained on ImageNet (transfer learning).
- In Phase A: only the final classifier head (fully connected layer) is fine-tuned; all other layers are frozen (feature extraction).
- In Phase B (optional / commented): the last convolutional block (layer4) can also be fine-tuned.

Main output:
- Trained model weights saved to `resnet18_yoga.pt`
"""
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import os
import sys
from torchmetrics import Accuracy, F1Score, AveragePrecision

# Allow importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Training_Helper_Funcs')))
from train_loop import train_loop, test_step



# --------------------------- Custom Dataset ---------------------------
class YogaCSVLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Dataset that loads images and labels from a CSV file.

        Args:
            csv_file (str): Path to CSV file containing image paths and labels.
            transform (callable, optional): Transformations to apply to the images.
        """
        print(f"Tha path: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label_idx']

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# -------------------------- Transforms --------------------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------------- Load CSV Splits --------------------------
train_csv = r"../Dataset_Divison/train_set.csv"
test_csv = r"../Dataset_Divison/val_set.csv"
val_csv = r"../Dataset_Divison/test_set.csv"

train_ds = YogaCSVLoader(train_csv, transform=transform)
val_ds = YogaCSVLoader(val_csv, transform=transform)
test_ds = YogaCSVLoader(test_csv, transform=transform)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

num_classes = len(pd.concat([train_ds.df, val_ds.df, test_ds.df])["label_idx"].unique())

# -------------------------- Model --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = models.resnet18(weights="IMAGENET1K_V1")
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, num_classes)
model.to(device)

# --------------------------
# Class weights (train only)
# --------------------------
classes = np.arange(num_classes)
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_ds.df["label_idx"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# # -------- Phase A: Fine-tune head only --------
#  Freeze everything
for p in model.parameters():
    p.requires_grad = False

#  Unfreeze ONLY the last layer (fc)
for p in model.fc.parameters():
    p.requires_grad = True
    
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
#opt = torch.optim.SGD(model.fc.parameters(), lr=1e-2)

print("Starting transfer learning (fine-tuning head only)...")
history, best_epoch = train_loop(
    model, train_dl, val_dl, opt, loss_fn,
    epochs=25, num_classes=num_classes
)

accuracy_score = Accuracy(task="multiclass", num_classes=num_classes).to(device)
f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
map_score = AveragePrecision(task="multiclass", num_classes=num_classes).to(device)

test_loss, test_acc, test_f1, test_map, test_ap_array = test_step(model, test_dl, loss_fn, device, accuracy_score, f1_score, map_score)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}| Test mAP: {test_map:.4f} | Test AP: {test_ap_array}")
test_results =  pd.DataFrame([{
    "Test Loss": test_loss,
    "Test Acc": test_acc,
    "Test F1": test_f1,
    "Test mAP": test_map,
    "Test AP": test_ap_array.tolist()
}])

test_results.to_csv("test_results_half_fine_tune.csv", index=False)
# -------- Phase B: Optional deeper fine-tuning --------
# Uncomment below for fine-tuning last conv block
# for p in model.layer4.parameters():
#     p.requires_grad_(True)
# opt = torch.optim.Adam([
#     {"params": model.layer4.parameters(), "lr": 1e-4},
#     {"params": model.fc.parameters(), "lr": 5e-4}
# ])
# print("Starting fine-tuning of last conv block (layer4)...")
# history, best_epoch = train_loop(
#     model, train_dl, val_dl, opt, loss_fn,
#     epochs=20, num_classes=num_classes
# )

# -------------------------- Save model --------------------------
torch.save(model.state_dict(), "resnet18_yoga_half_fine_tune.pt")
print("Model saved to resnet18_yoga_half_fine_tune.pt")

# Save history
history_df = pd.DataFrame(history)
history_df.to_csv("resnet_training_history_half_fine_tune.csv", index=False)
print("Training history saved to training_history_half_fine_tune.csv")
num_tune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_total_params = sum(p.numel() for p in model.parameters())

print(f"Number of fine-tuned parameters: {num_tune_params}")
print(f"Total number of parameters: {num_total_params}")
print(f"Percentage fine-tuned: {100 * num_tune_params / num_total_params:.2f}%")