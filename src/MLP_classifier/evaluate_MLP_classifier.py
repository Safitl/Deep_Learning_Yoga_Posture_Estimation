# evaluate_keypoints_classifier.py
"""
Evaluates a pre-trained keypoints classifier on the hold-out test set.

Workflow:
1.  Loads the training data ONLY to fit a StandardScaler for consistent preprocessing.
2.  Loads the test data and applies the fitted scaler.
3.  Loads saved hyperparameters and reconstructs the model architecture.
4.  Loads the saved best model weights.
5.  Calculates and prints performance metrics (Accuracy, F1, mAP).
6.  Generates and saves Precision-Recall (P-R) curves for the test set.
"""
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import MulticlassAveragePrecision
from itertools import cycle

# --- Configuration ---
RESULTS_DIR = "Pose_Keypoints/Results_kaggle_final"
BASE_DATA_PATH = "Pose_Keypoints/Datasets_CSV/Keypoints_with_conf_CSVs"
os.makedirs(os.path.join(RESULTS_DIR, "Evaluation_Figures"), exist_ok=True)

TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, "train_set_updated_kp_conf.csv")
TEST_CSV_PATH = os.path.join(BASE_DATA_PATH, "test_set_updated_kp_conf.csv")

WEIGHTS_PATH = os.path.join(RESULTS_DIR, "best_model_weights.pth")
PARAMS_PATH = os.path.join(RESULTS_DIR, "hyperparameters.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (Must match the one used for training) ---
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
def plot_pr_curves(y_true, y_scores, class_names, results_dir):
    """Generates and saves per-class and micro-averaged Precision-Recall curves."""
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    ap_micro = average_precision_score(y_true_bin, y_scores, average="micro")
    plt.figure(figsize=(10, 7))
    plt.step(recall_micro, precision_micro, where='post', label=f'Micro-average P-R curve (AP = {ap_micro:0.2f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.title('Micro-Averaged Precision-Recall Curve (Test Set)'); plt.legend(loc="best"); plt.grid(True)
    plt.savefig(os.path.join(results_dir, "Evaluation_Figures", 'precision_recall_meanAP.png'))
    plt.close()

if __name__ == "__main__":
    # 1. Load data and scaler from training phase
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    keypoint_columns = [col for col in train_df.columns if col.startswith("kp_")]
    scaler = StandardScaler().fit(train_df[keypoint_columns])
    test_df[keypoint_columns] = scaler.transform(test_df[keypoint_columns])
    class_names = train_df.sort_values('label_idx')['label_str'].unique()
    
    X_test = torch.tensor(test_df[keypoint_columns].values, dtype=torch.float32)
    y_test = torch.tensor(test_df["label_idx"].values, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    # 2. Load hyperparameters and reconstruct model
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    
    model = KaggleYogaClassifier(input_dim=params['input_dim'], num_classes=params['num_classes']).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Evaluate the model on the test set
    print("🚀 Evaluating model on the test set...")
    acc_metric = Accuracy(task="multiclass", num_classes=params['num_classes']).to(DEVICE)
    f1_metric = F1Score(task="multiclass", num_classes=params['num_classes']).to(DEVICE)
    ap_metric = MulticlassAveragePrecision(num_classes=params['num_classes'], average="macro").to(DEVICE)
    y_true_list, y_score_list = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            acc_metric.update(logits, y); f1_metric.update(logits, y); ap_metric.update(logits, y)
            y_true_list.append(y.cpu())
            y_score_list.append(torch.softmax(logits, dim=1).cpu())

    print("\n--- Test Set Final Metrics ---")
    print(f"Accuracy: {acc_metric.compute().item():.4f}")
    print(f"F1 Score (Macro): {f1_metric.compute().item():.4f}")
    print(f"mAP (mean Average Precision): {ap_metric.compute().item():.4f}")
    print("----------------------------")

    # 4. Generate and save plots
    y_true, y_scores = torch.cat(y_true_list).numpy(), torch.cat(y_score_list).numpy()
    plot_pr_curves(y_true, y_scores, class_names, RESULTS_DIR)
    print(f"📈 Precision-Recall curves saved to '{os.path.join(RESULTS_DIR, 'Evaluation_Figures')}'")