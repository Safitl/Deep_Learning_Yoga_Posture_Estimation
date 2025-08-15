"""
This script trains the MultiTokenTransformer model with a fixed set of hyperparameters.

It orchestrates the entire process, including:
1.  Loading the training, validation, and test datasets from CSV files.
2.  Calculating normalization statistics (mean/std) from the training data to
    prevent data leakage.
3.  Creating PyTorch DataLoaders with on-the-fly normalization.
4.  Calculating class weights to address class imbalance in the training set.
5.  Initializing the MultiTokenTransformer model with a specific configuration.
6.  Setting up the AdamW optimizer and a CosineAnnealingLR learning rate scheduler.
7.  Launching the main training loop to train and validate the model.
"""
import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# Import custom modules from the project
from MultiTokenTransformer import MultiTokenTransformer
from transformer_train_loop import train_loop

# --- 1. Data Loading & Preparation ---

def make_tensor_ds(csv_path: str, kp_mu: torch.Tensor, kp_std: torch.Tensor,
                   cnn_mu: torch.Tensor, cnn_std: torch.Tensor) -> TensorDataset:
    """
    Loads data from a CSV, converts it to tensors, and applies Z-score normalization.

    Args:
        csv_path (str): The path to the input CSV file.
        kp_mu, kp_std (torch.Tensor): Mean and std for keypoint normalization,
                                      calculated from the training set.
        cnn_mu, cnn_std (torch.Tensor): Mean and std for CNN feature normalization,
                                        calculated from the training set.

    Returns:
        TensorDataset: A PyTorch dataset of normalized (kp, cnn, y) tensors.
    """
    df = pd.read_csv(csv_path)

    # Extract features and labels as tensors
    kp = torch.tensor(df[KP_COLS].values, dtype=torch.float32)
    cnn = torch.tensor(df[CNN_COLS].values, dtype=torch.float32)
    y = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # Apply Z-score normalization using the pre-computed training statistics
    kp = (kp - kp_mu) / kp_std
    cnn = (cnn - cnn_mu) / cnn_std

    return TensorDataset(kp, cnn, y)

# Load raw dataframes from CSV files
train_raw = pd.read_csv("Transformer/Datasets_CSV/datasets_half_fine_tune/train_set_half_fine_tune_kp_conf.csv")
val_raw = pd.read_csv("Transformer/Datasets_CSV/datasets_half_fine_tune/test_set_half_fine_tune_kp_conf.csv")
test_raw = pd.read_csv("Transformer/Datasets_CSV/datasets_half_fine_tune/val_set_half_fine_tune_kp_conf.csv")

# Automatically identify keypoint and CNN feature columns
KP_COLS = [c for c in train_raw.columns if c.startswith("kp_")]
CNN_COLS = [c for c in train_raw.columns if c.startswith("cnn_")]


# --- 2. Normalization & DataLoader Setup ---

# Calculate normalization statistics (mean and std) from the TRAINING data ONLY.
kp_mu = torch.tensor(train_raw[KP_COLS].mean().values, dtype=torch.float32)
kp_std = torch.tensor(train_raw[KP_COLS].std().values + 1e-8, dtype=torch.float32)
cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
cnn_std = torch.tensor(train_raw[CNN_COLS].std().values + 1e-8, dtype=torch.float32)

# Create DataLoaders for each split
BATCH_SIZE = 64
train_dl = DataLoader(make_tensor_ds("Transformer/Datasets_CSV/datasets_half_fine_tune/train_set_half_fine_tune_kp_conf.csv", kp_mu, kp_std, cnn_mu, cnn_std), batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(make_tensor_ds("Transformer/Datasets_CSV/datasets_half_fine_tune/test_set_half_fine_tune_kp_conf.csv", kp_mu, kp_std, cnn_mu, cnn_std), batch_size=BATCH_SIZE)
test_dl = DataLoader(make_tensor_ds("Transformer/Datasets_CSV/datasets_half_fine_tune/val_set_half_fine_tune_kp_conf.csv", kp_mu, kp_std, cnn_mu, cnn_std), batch_size=BATCH_SIZE)


# --- 3. Model & Training Configuration ---

# Initialize the model with a fixed hyperparameter configuration
model = MultiTokenTransformer(
    d_model=256,
    nhead=4,
    n_layers=2,
    dim_ff=2 * 256,
    num_classes=47
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Configure optimizer and learning rate scheduler
EPOCHS = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Calculate class weights to counteract class imbalance during training
classes = np.arange(47)
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_raw["label_idx"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

# Initialize the loss function with the calculated class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


# --- 4. Training Execution ---

print("🚀 Starting training for the MultiTokenTransformer...")
# Call the main training loop to start the training and validation process
history, best_epoch = train_loop(
    model=model,
    trainloader=train_dl,
    testloader=val_dl,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS,
    num_classes=47,
    verbose=True
)
print("✅ Training finished.")