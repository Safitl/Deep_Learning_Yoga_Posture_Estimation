"""
Trains a simple MLP classifier on yoga pose keypoints and saves the resulting artifacts.

Workflow:
1.  Loads and preprocesses the training and validation data.
2.  Defines and initializes the MLP model, optimizer, and loss function.
3.  Runs a training and validation loop for a fixed number of epochs.
4.  Saves the best model weights based on validation accuracy.
5.  Saves the hyperparameters and plots of the training curves.
"""
import os
import json
import copy
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# --- Configuration ---
RESULTS_DIR = "Pose_Keypoints/Results_kaggle_final"
BASE_DATA_PATH = "Pose_Keypoints/Datasets_CSV/Keypoints_with_conf_CSVs"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, "train_set_updated_kp_conf.csv")
VAL_CSV_PATH = os.path.join(BASE_DATA_PATH, "val_set_updated_kp_conf.csv")

WEIGHTS_SAVE_PATH = os.path.join(RESULTS_DIR, "best_model_weights.pth")
PARAMS_SAVE_PATH = os.path.join(RESULTS_DIR, "hyperparameters.json")
SCALER_SAVE_PATH = os.path.join(RESULTS_DIR, "scaler.json")

LEARNING_RATE, BATCH_SIZE, EPOCHS = 0.001, 32, 30
OPTIMIZER_NAME = "Adam"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class KaggleYogaClassifier(nn.Module):
    """A simple MLP classifier with two hidden layers."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.outlayer = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x)); x = self.dropout(x)
        x = self.relu(self.layer2(x)); return self.outlayer(x)

# --- Helper Functions ---
def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_training_curves(history, results_dir):
    """Plots and saves training & validation accuracy and loss curves."""
    plt.figure(figsize=(10, 5)); plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy vs. Epochs'); plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(results_dir, 'training_accuracy_plot.png')); plt.close()
    
    plt.figure(figsize=(10, 5)); plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss vs. Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.savefig(os.path.join(results_dir, 'training_loss_plot.png')); plt.close()

if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    keypoint_columns = [col for col in train_df.columns if col.startswith("kp_")]
    input_dim = len(keypoint_columns)

    scaler = StandardScaler().fit(train_df[keypoint_columns])
    train_df[keypoint_columns] = scaler.transform(train_df[keypoint_columns])
    val_df[keypoint_columns] = scaler.transform(val_df[keypoint_columns])

    X_train, y_train = torch.tensor(train_df[keypoint_columns].values, dtype=torch.float32), torch.tensor(train_df["label_idx"].values, dtype=torch.long)
    X_val, y_val = torch.tensor(val_df[keypoint_columns].values, dtype=torch.float32), torch.tensor(val_df["label_idx"].values, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    num_classes = len(train_df['label_idx'].unique())
    
    model = KaggleYogaClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_model_wts = 0.0, None

    print(f"🚀 Training model for {EPOCHS} epochs on {DEVICE}...")
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch.to(DEVICE)); loss = loss_fn(logits, y_batch.to(DEVICE))
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_correct += (torch.argmax(logits, dim=1) == y_batch.to(DEVICE)).sum().item()
            train_total += y_batch.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(DEVICE)); loss = loss_fn(logits, y_batch.to(DEVICE))
                val_loss += loss.item()
                val_correct += (torch.argmax(logits, dim=1) == y_batch.to(DEVICE)).sum().item()
                val_total += y_batch.size(0)

        history["train_acc"].append(train_correct / train_total); history["val_acc"].append(val_correct / val_total)
        history["train_loss"].append(train_loss / len(train_loader)); history["val_loss"].append(val_loss / len(val_loader))

        if history["val_acc"][-1] > best_val_acc:
            best_val_acc = history["val_acc"][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    
    print("\n✅ Training complete.")
    torch.save(best_model_wts, WEIGHTS_SAVE_PATH)
    print(f"💾 Best model weights saved to '{WEIGHTS_SAVE_PATH}'")

    hyperparams = {"input_dim": input_dim, "num_classes": num_classes, "Optimizer": OPTIMIZER_NAME, 
                   "Learning Rate": LEARNING_RATE, "Epochs": EPOCHS, "Batch Size": BATCH_SIZE, 
                   "Num of Params": count_parameters(model)}
    with open(PARAMS_SAVE_PATH, 'w') as f: json.dump(hyperparams, f, indent=4)
    print(f"💾 Hyperparameters saved to '{PARAMS_SAVE_PATH}'")

    plot_training_curves(history, RESULTS_DIR)
    print(f"📈 Training curves saved to '{RESULTS_DIR}'")