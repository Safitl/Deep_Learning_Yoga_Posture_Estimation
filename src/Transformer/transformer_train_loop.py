# transformer_train_loop.py
"""
Provides a refactored, robust, and reusable training and evaluation pipeline
for two-input PyTorch models, such as the TwoTokenTransformer or MultiTokenTransformer.

This module encapsulates the logic for:
- A single training epoch (`train_step`).
- A single validation/testing epoch (`test_step`).
- The main orchestration loop (`train_loop`) that runs for multiple epochs,
  tracks metrics, displays a progress bar, and saves the best performing model
  based on validation accuracy.

It uses `torchmetrics` for accurate metric calculation and `tqdm` for a user-friendly
progress bar.
"""
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm

def _reset_metrics(*metrics):
    """
    A helper function to reset the internal state of one or more torchmetrics objects.
    This is called at the beginning of each epoch to ensure fresh metric calculation.

    Args:
        *metrics: A variable number of torchmetrics objects (e.g., Accuracy, F1Score).
    """
    for m in metrics:
        m.reset()

def train_step(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module, device: torch.device,
               accuracy_score: Accuracy, f1_score: F1Score) -> tuple[float, float, float]:
    """
    Performs a single training epoch on the entire training dataset.

    This involves setting the model to training mode, iterating over all batches,
    performing forward and backward passes, and updating model weights.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to run on.
        accuracy_score (Accuracy): A torchmetrics Accuracy object.
        f1_score (F1Score): A torchmetrics F1Score object.

    Returns:
        tuple[float, float, float]: A tuple containing the average loss, accuracy,
                                    and F1 score for the epoch.
    """
    model.train()
    running_loss = 0.0
    _reset_metrics(accuracy_score, f1_score) # Reset metrics at the start of the epoch

    # Iterate over data batches
    for kp, cnn, y in dataloader:
        kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)

        # 1. Forward pass: Compute predicted outputs by passing inputs to the model
        logits = model(kp, cnn)
        loss = loss_fn(logits, y)

        # 2. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the total loss for the epoch (weighted by batch size)
        running_loss += loss.item() * y.size(0)
        # Update metrics with the results from the current batch
        accuracy_score.update(logits, y)
        f1_score.update(logits, y)

    # Calculate average metrics over the entire dataset for the epoch
    n = len(dataloader.dataset)
    epoch_loss = running_loss / n
    epoch_acc = accuracy_score.compute().item()
    epoch_f1 = f1_score.compute().item()

    return epoch_loss, epoch_acc, epoch_f1

@torch.no_grad()
def test_step(model: torch.nn.Module, dataloader: DataLoader, loss_fn: torch.nn.Module,
              device: torch.device, accuracy_score: Accuracy, f1_score: F1Score) -> tuple[float, float, float]:
    """
    Performs a single validation/test epoch on the entire dataset.

    Disables gradient calculations for efficiency.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (DataLoader): DataLoader for the validation/test data.
        loss_fn (torch.nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to run on.
        accuracy_score (Accuracy): A torchmetrics Accuracy object.
        f1_score (F1Score): A torchmetrics F1Score object.

    Returns:
        tuple[float, float, float]: A tuple containing the average loss, accuracy,
                                    and F1 score for the epoch.
    """
    model.eval()
    running_loss = 0.0
    _reset_metrics(accuracy_score, f1_score)

    for kp, cnn, y in dataloader:
        kp, cnn, y = kp.to(device), cnn.to(device), y.to(device)

        # Forward pass
        logits = model(kp, cnn)
        loss = loss_fn(logits, y)

        # Accumulate loss and update metrics
        running_loss += loss.item() * y.size(0)
        accuracy_score.update(logits, y)
        f1_score.update(logits, y)

    # Calculate final metrics for the epoch
    n = len(dataloader.dataset)
    epoch_loss = running_loss / n
    epoch_acc = accuracy_score.compute().item()
    epoch_f1 = f1_score.compute().item()

    return epoch_loss, epoch_acc, epoch_f1

def train_loop(model: torch.nn.Module,
               trainloader: DataLoader,
               testloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               epochs: int,
               num_classes: int,
               save_path: str = "best_weights_trans.pth",
               verbose: bool = True) -> tuple[dict, int]:
    """
    The main training and validation loop.

    Orchestrates the training process over multiple epochs, tracks performance,
    and saves the best model checkpoint based on validation accuracy.

    Args:
        model (torch.nn.Module): The model to train.
        trainloader (DataLoader): DataLoader for the training set.
        testloader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (torch.nn.Module): The loss function.
        epochs (int): The total number of epochs to train.
        num_classes (int): The number of classes for metric calculation.
        save_path (str): The file path where the best model weights will be saved.
        verbose (bool): If True, prints progress and results.

    Returns:
        tuple[dict, int]: A tuple containing:
            - A history dictionary with metrics for each epoch.
            - The epoch number (0-indexed) that achieved the best validation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize metric objects
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

    # Dictionary to store metrics over time
    history = {k: [] for k in
               ("train_loss", "train_accuracy", "train_f1",
                "test_loss", "test_accuracy", "test_f1")}

    best_acc = 0.0
    best_epoch = -1

    progress_bar = tqdm(range(epochs), desc="Initializing...", disable=not verbose)

    for epoch in progress_bar:
        # Perform one epoch of training and one epoch of validation
        train_loss, train_acc, train_f1 = train_step(model, trainloader, optimizer, loss_fn, device, accuracy, f1)
        val_loss, val_acc, val_f1 = test_step(model, testloader, loss_fn, device, accuracy, f1)

        # Record metrics in the history dictionary
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["test_loss"].append(val_loss)
        history["test_accuracy"].append(val_acc)
        history["test_f1"].append(val_f1)

        # Checkpoint: Save the model if validation accuracy has improved
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path) # Save to the specified path

        # Update the progress bar with the latest metrics
        progress_bar.set_description(
            f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.3f} | Best: {best_acc:.3f} @ Ep {best_epoch+1}"
        )

    if verbose:
        print(f"\n✅ Training complete. Best validation accuracy = {best_acc:.4f} at epoch {best_epoch+1}.")
        print(f"   - Best model weights saved to: '{save_path}'")

    return history, best_epoch