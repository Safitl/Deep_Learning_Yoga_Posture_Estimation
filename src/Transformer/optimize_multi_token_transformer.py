# optimize_multi_token_transformer.py
"""
Performs hyperparameter optimization (HPO) for the MultiTokenTransformer model
using the Optuna framework.

This script is designed to be robust and resumable. It systematically searches
for the best combination of hyperparameters (e.g., model dimensions, learning rate)
by running multiple training trials.

Key Responsibilities:
1.  **Setup & Configuration**: Defines paths, seeds, and HPO study parameters.
2.  **Data Loading**: Loads pre-processed train/validation CSVs containing keypoint
    and CNN features.
3.  **HPO Trial Definition**: The `objective` function defines a single training and
    evaluation run for a given set of hyperparameters suggested by Optuna.
4.  **Pruning**: Integrates with Optuna's MedianPruner to automatically stop
    unpromising trials early, saving computational resources.
5.  **Persistence**: Saves the study's progress to an SQLite database, allowing the
    optimization process to be paused and resumed.
6.  **Artifact Management**:
    - Each trial saves its best model weights to a temporary file.
    - After the study completes, the script identifies the best trial and copies
      its weights to a final destination.
    - It saves the best hyperparameters and training history as JSON files.
    - It cleans up all temporary weight files.
"""
import os
import json
import math
import random
import shutil
import glob
import numpy as np
import pandas as pd
from functools import partial

import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

from MultiTokenTransformer import MultiTokenTransformer
from transformer_train_loop import train_loop

# --- CONFIGURATION & GLOBAL CONSTANTS ---
RESULTS_DIR = "Transformer/Results/MultiToken_HPO_Final"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 47
N_TRIALS = 20  # Total number of HPO trials to run
EPOCHS_PER_TRIAL = 15 # Number of epochs to train for within a single trial

# Define final output paths for the best trial's artifacts
BEST_MODEL_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "best_weights_multi_token.pth")
HISTORY_PATH = os.path.join(RESULTS_DIR, "best_trial_history_multi_token.json")
PARAMS_PATH = os.path.join(RESULTS_DIR, "best_hyperparameters_multi_token.json")

# Optuna study configuration for persistence
STUDY_NAME = "multi-token-transformer-hpo"
STORAGE_NAME = f"sqlite:///{os.path.join(RESULTS_DIR, f'{STUDY_NAME}.db')}"


def setup_seeds():
    """Sets random seeds for PyTorch, NumPy, and random for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

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
        TensorDataset: A dataset containing normalized keypoint tensors, cnn tensors,
                       and label tensors.
    """
    kp = torch.tensor(df[kp_cols].values, dtype=torch.float32)
    cnn = torch.tensor(df[cnn_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label_idx"].values, dtype=torch.long)

    # Apply Z-score normalization using pre-computed training set statistics
    kp = (kp - kp_mu) / kp_std
    cnn = (cnn - cnn_mu) / cnn_std

    return TensorDataset(kp, cnn, y)

# --------------------------
# 1. Optuna Objective Function
# --------------------------
def objective(trial: optuna.Trial, train_ds: TensorDataset, val_ds: TensorDataset,
              class_weights: torch.Tensor, kp_dim: int, cnn_dim: int, results_dir: str) -> float:
    """
    Defines and executes a single HPO trial for Optuna.

    This function is called by Optuna for each trial. It:
    1.  Suggests hyperparameters from the defined search space.
    2.  Builds the model, optimizer, and data loaders.
    3.  Runs the training loop.
    4.  Reports intermediate validation accuracy to the pruner.
    5.  Saves trial-specific artifacts (weights path, history) as user attributes.
    6.  Returns the final best validation accuracy for the trial.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        train_ds, val_ds (TensorDataset): The training and validation datasets.
        class_weights (torch.Tensor): Class weights for the loss function.
        kp_dim, cnn_dim (int): Dimensionality of the input features.
        results_dir (str): Directory to save temporary trial artifacts.

    Returns:
        float: The best validation accuracy achieved during the trial.
    """
    # Each trial saves its own temporary weights file to avoid race conditions.
    trial_weights_path = os.path.join(results_dir, f"trial_{trial.number}_weights.pth")

    # --- Hyperparameter Search Space ---
    d_model = trial.suggest_categorical("d_model", [128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    # Constraint: d_model must be divisible by nhead for multi-head attention.
    if d_model % nhead != 0:
        raise TrialPruned(f"Invalid combo: d_model={d_model}, nhead={nhead} (not divisible).")

    n_layers = trial.suggest_int("n_layers", 1, 4)
    dim_ff_multiplier = trial.suggest_categorical("dim_ff_multiplier", [2, 4])
    dim_ff = d_model * dim_ff_multiplier
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # --- Setup Model, Dataloaders, and Optimizer ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = MultiTokenTransformer(
        kp_dim=kp_dim, cnn_dim=cnn_dim, d_model=d_model, nhead=nhead,
        n_layers=n_layers, dim_ff=dim_ff, dropout=dropout, num_classes=NUM_CLASSES
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --- Run Training ---
    history, best_epoch = train_loop(
        model, train_loader, val_loader, opt, loss_fn,
        epochs=EPOCHS_PER_TRIAL, num_classes=NUM_CLASSES,
        save_path=trial_weights_path, # Pass the unique path for this trial
        verbose=False
    )

    # --- Report to Pruner and Handle NaN/Inf ---
    val_accs = history.get("test_accuracy", [])
    if not val_accs or any(a is None or math.isnan(a) or math.isinf(a) for a in val_accs):
        raise TrialPruned("Trial resulted in NaN, Inf, or empty validation accuracy.")

    for ep, acc in enumerate(val_accs):
        trial.report(float(acc), step=ep)
        if trial.should_prune():
            raise TrialPruned(f"Pruned at epoch {ep} with val_acc={acc:.4f}")

    # --- Save Trial Results as User Attributes ---
    # Storing large objects like weights directly in the study can bloat the database.
    # Instead, we save them to a file and store the *path* to that file.
    best_val_acc = float(max(val_accs))
    trial.set_user_attr("history", history) # History JSON is small and safe to store
    trial.set_user_attr("best_epoch", int(best_epoch) if best_epoch is not None else -1)
    trial.set_user_attr("weights_path", trial_weights_path) # Store the path

    return best_val_acc

def save_best_results(study: optuna.study.Study):
    """
    Analyzes the completed Optuna study and saves the artifacts from the best trial.

    Args:
        study (optuna.study.Study): The completed Optuna study object.
    """
    print("\n" + "="*50)
    print("ANALYZING STUDY AND SAVING BEST RESULTS...")
    try:
        best_trial = study.best_trial
        print(f"🏆 Best Trial Found: #{best_trial.number}")
        print(f"  > Value (Best Val Accuracy): {best_trial.value:.4f}")
        print("  > Params:")
        for key, value in best_trial.params.items():
            print(f"    - {key}: {value}")
        print("="*50 + "\n")

        # Save history and hyperparameters JSON files
        with open(HISTORY_PATH, "w") as f:
            json.dump(best_trial.user_attrs["history"], f, indent=2)
        with open(PARAMS_PATH, "w") as f:
            json.dump(best_trial.params, f, indent=2)

        # Copy the best trial's weight file from its temporary path to the final destination
        best_weights_file_path = best_trial.user_attrs["weights_path"]
        if os.path.exists(best_weights_file_path):
            shutil.copy(best_weights_file_path, BEST_MODEL_WEIGHTS_PATH)
            print(f"✅ Saved best trial history to '{HISTORY_PATH}'")
            print(f"✅ Saved best hyperparameters to '{PARAMS_PATH}'")
            print(f"💾 Saved final best model weights to '{BEST_MODEL_WEIGHTS_PATH}'")
        else:
            print(f"❌ Error: Could not find the best trial's weight file at '{best_weights_file_path}'")

    except ValueError:
        print("❌ No completed trials found in the study. Nothing to save.")
    except KeyError:
        print("❌ Could not find 'weights_path' attribute. This might happen if the best trial was from a previous run.")

# --------------------------
# 2. Main Execution Block
# --------------------------
if __name__ == "__main__":
    setup_seeds()

    print(f"📂 Ensuring results directory exists: '{RESULTS_DIR}'")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Data Loading and Preparation ---
    print("🔄 Loading and preparing data splits...")
    try:
        train_csv_path = "Transformer/Datasets_CSV/datasets_half_fine_tune/train_set_half_fine_tune_kp_conf.csv"
        val_csv_path = "Transformer/Datasets_CSV/datasets_half_fine_tune/val_set_half_fine_tune_kp_conf.csv"
        train_raw = pd.read_csv(train_csv_path)
        val_raw = pd.read_csv(val_csv_path)
    except FileNotFoundError as e:
        print(f"❌ Critical Error: Data file not found. {e}")
        exit()

    # Identify feature columns
    KP_COLS = [c for c in train_raw.columns if c.startswith("kp_")]
    CNN_COLS = [c for c in train_raw.columns if c.startswith("cnn_")]
    print(f"✅ Found {len(KP_COLS)} keypoint features and {len(CNN_COLS)} CNN features.")

    # Clean data (coerce to numeric and drop rows with NaNs)
    for df in (train_raw, val_raw):
        df[KP_COLS] = df[KP_COLS].apply(pd.to_numeric, errors="coerce")
        df[CNN_COLS] = df[CNN_COLS].apply(pd.to_numeric, errors="coerce")
        df.dropna(subset=KP_COLS + CNN_COLS + ["label_idx"], inplace=True)

    # Calculate normalization statistics from the training set ONLY
    kp_mu = torch.tensor(train_raw[KP_COLS].mean().values, dtype=torch.float32)
    kp_std = torch.tensor(train_raw[KP_COLS].std().values + 1e-8, dtype=torch.float32)
    cnn_mu = torch.tensor(train_raw[CNN_COLS].mean().values, dtype=torch.float32)
    cnn_std = torch.tensor(train_raw[CNN_COLS].std().values + 1e-8, dtype=torch.float32)

    # Create TensorDatasets
    train_ds = make_tensor_ds(train_raw, KP_COLS, CNN_COLS, kp_mu, kp_std, cnn_mu, cnn_std)
    val_ds = make_tensor_ds(val_raw, KP_COLS, CNN_COLS, kp_mu, kp_std, cnn_mu, cnn_std)

    # Calculate class weights for handling class imbalance
    classes = np.arange(NUM_CLASSES)
    class_weights_np = compute_class_weight("balanced", classes=classes, y=train_raw["label_idx"].values)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)
    print("✅ Data loading, normalization, and class weight calculation complete.\n")

    # --- HPO Study Execution ---
    print(f"🚀 Starting/Resuming Optuna study '{STUDY_NAME}'...")
    print(f"   - Storage: {STORAGE_NAME}")
    print(f"   - Number of trials to run: {N_TRIALS}")

    sampler = TPESampler(seed=SEED, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

    # Create the study. `load_if_exists=True` makes it resumable.
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE_NAME,
        direction="maximize", sampler=sampler, pruner=pruner,
        load_if_exists=True
    )

    # Use a lambda function to pass the static arguments to the objective function
    objective_func = partial(
        objective,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        kp_dim=len(KP_COLS),
        cnn_dim=len(CNN_COLS),
        results_dir=RESULTS_DIR
    )

    try:
        study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n🛑 Study interrupted by user. Saving best results so far...")
    finally:
        # This block runs whether the study completes or is interrupted.
        save_best_results(study)

        # Clean up all temporary per-trial weight files
        print("🧹 Cleaning up temporary trial weight files...")
        cleaned_count = 0
        for f in glob.glob(os.path.join(RESULTS_DIR, "trial_*_weights.pth")):
            os.remove(f)
            cleaned_count += 1
        print(f"   - Removed {cleaned_count} temporary files.")