"""
Evaluates the best-performing MultiTokenTransformer model on the hold-out test set.

This script provides the final, unbiased performance assessment of the model trained
during the hyperparameter optimization (HPO) process.

Workflow:
1.  Loads the best hyperparameters from the JSON file saved by the HPO script.
2.  Loads the saved model weights (`.pth` file).
3.  Loads the training data *only* to compute the normalization statistics (mean/std),
    ensuring the test data is transformed identically to the training data.
4.  Loads the test data, applies the normalization, and prepares it for evaluation.
5.  Initializes the model with the best hyperparameters and loads the saved weights.
6.  Runs inference on the test set to gather predictions.
7.  Generates and prints a detailed classification report (precision, recall, F1-score).
8.  Computes and plots a confusion matrix, saving it as a PNG file.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from MultiTokenTransformer import MultiTokenTransformer

# --- Configuration ---
# Define paths for loading artifacts and saving results
RESULTS_DIR = "Transformer/Results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "Figures")
HP_PATH = os.path.join(RESULTS_DIR, "best_hyperparameters_multi_token.json")
WEIGHTS_PATH = os.path.join(RESULTS_DIR, "best_weights_multi_token.pth")

# Define paths to the datasets
TRAIN_CSV_PATH = "Transformer/Datasets_CSV/datasets_half_fine_tune/train_set_half_fine_tune_kp_conf.csv"
TEST_CSV_PATH = "Transformer/Datasets_CSV/datasets_half_fine_tune/test_set_half_fine_tune_kp_conf.csv"

# Global constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 47


def make_tensor_ds(df: pd.DataFrame, kp_cols: list, cnn_cols: list,
                   kp_mu: torch.Tensor, kp_std: torch.Tensor,
                   cnn_mu: torch.Tensor, cnn_std: torch.Tensor) -> TensorDataset:
    """
    Creates a normalized PyTorch TensorDataset from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        kp_cols (list): List of keypoint column names.
        cnn_cols (list): List of CNN feature column names.
        kp_mu, kp_std (torch.Tensor): Mean and std for keypoint normalization.
        cnn_mu, cnn_std (torch.Tensor): Mean and std for CNN normalization.

    Returns:
        TensorDataset: A dataset containing normalized tensors.
    """
    kp = torch.tensor(df[kp_cols].values, dtype=torch.float32)
    cnn = torch.tensor(df[cnn_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # Apply Z-score normalization using pre-computed training set statistics
    kp = (kp - kp_mu) / kp_std
    cnn = (cnn - cnn_mu) / cnn_std

    return TensorDataset(kp, cnn, y)


def main():
    """
    Main function to orchestrate the model evaluation process.
    """
    print(f"📂 Ensuring figures directory exists: '{FIGURES_DIR}'")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- 1. Load Hyperparameters ---
    print(f"🔄 Loading best hyperparameters from '{HP_PATH}'...")
    try:
        with open(HP_PATH, 'r') as f:
            params = json.load(f)
        print("✅ Hyperparameters loaded successfully:", params)
    except FileNotFoundError:
        print(f"❌ Error: Hyperparameter file not found at '{HP_PATH}'. Exiting.")
        return

    # --- 2. Load Data and Preprocess ---
    print("🔄 Loading and preparing data...")
    # Load training data ONLY to get the correct normalization statistics.
    # This prevents data leakage from the test set.
    train_raw = pd.read_csv(TRAIN_CSV_PATH)
    KP_COLS = [c for c in train_raw.columns if c.startswith("kp_")]
    CNN_COLS = [c for c in train_raw.columns if c.startswith("cnn_")]

    kp_mu = torch.tensor(train_raw[KP_COLS].mean().values, dtype=torch.float32)
    kp_std = torch.tensor(train_raw[KP_COLS].std().values + 1e-8, dtype=torch.float32)
    cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
    cnn_std = torch.tensor(train_raw[CNN_COLS].std().values + 1e-8, dtype=torch.float32)

    # Now, load and process the actual test data using the training statistics
    test_raw = pd.read_csv(TEST_CSV_PATH)
    test_ds = make_tensor_ds(test_raw, KP_COLS, CNN_COLS, kp_mu, kp_std, cnn_mu, cnn_std)
    test_loader = DataLoader(test_ds, batch_size=params.get('batch_size', 32), shuffle=False)
    print("✅ Test data prepared.")

    # --- 3. Initialize Model and Load Weights ---
    print(f"🔄 Initializing model and loading weights from '{WEIGHTS_PATH}'...")
    # Re-create the model architecture with the exact same hyperparameters
    model = MultiTokenTransformer(
        kp_dim=len(KP_COLS),
        cnn_dim=len(CNN_COLS),
        d_model=params['d_model'],
        nhead=params['nhead'],
        n_layers=params['n_layers'],
        dim_ff=params['d_model'] * params['dim_ff_multiplier'],
        dropout=params['dropout'],
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    try:
        # Load the saved state dictionary into the model
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        # Set the model to evaluation mode (disables dropout, etc.)
        model.eval()
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model weights file not found at '{WEIGHTS_PATH}'. Exiting.")
        return
    except RuntimeError as e:
        print(f"❌ Error: State dict mismatch. The model architecture might not match the saved weights. {e}")
        return


    # --- 4. Evaluate on Test Set ---
    print("🚀 Evaluating on the test set...")
    all_preds = []
    all_labels = []
    # Use torch.no_grad() to disable gradient calculations for efficiency
    with torch.no_grad():
        for kp, cnn, y in test_loader:
            kp, cnn = kp.to(DEVICE), cnn.to(DEVICE)
            logits = model(kp, cnn)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # --- 5. Print Metrics and Plot Confusion Matrix ---
    print("\n" + "="*50)
    print("          Test Set Classification Report")
    print("="*50)
    # Generate a detailed report with precision, recall, and F1-score for each class
    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)])
    print(report)

    # Generate and save the confusion matrix plot
    print("🔄 Generating confusion matrix plot...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title('Confusion Matrix - Test Set', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Save the plot to the figures directory
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix_test_set.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close() # Free up memory
    print(f"✅ Confusion matrix saved to '{cm_path}'")


if __name__ == "__main__":
    main()