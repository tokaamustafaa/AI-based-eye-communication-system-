"""
utils.py — Shared helpers: early stopping, checkpointing, metric computation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model weights automatically.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None
        self.should_stop = False

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Call once per epoch. Returns True when training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy state dict so it is not mutated by future updates
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"  [EarlyStopping] val_loss improved to {val_loss:.6f}. Saving best weights.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("  [EarlyStopping] Patience exhausted. Stopping training.")
        return self.should_stop

    def restore_best_weights(self, model: torch.nn.Module):
        """Load the best-performing weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print(f"  [EarlyStopping] Best weights restored (val_loss={self.best_loss:.6f}).")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    path: str):
    """Save model + optimizer state to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  [Checkpoint] Saved → {path}  (epoch={epoch}, val_loss={val_loss:.6f})")


def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: torch.device | None = None):
    """
    Load checkpoint from disk.
    Returns (epoch, val_loss). Mutates model (and optionally optimizer) in-place.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"  [Checkpoint] Loaded ← {path}  (epoch={epoch}, val_loss={val_loss:.6f})")
    return epoch, val_loss


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    preds, targets: (N, 2) float arrays with normalized [0,1] x/y values.

    Returns dict with:
        mae_x, mae_y, mae_overall  — Mean Absolute Error
        rmse_x, rmse_y, rmse_overall — Root Mean Squared Error
        euclidean_mean, euclidean_std  — Per-sample Euclidean distance stats
    """
    assert preds.shape == targets.shape, "Shape mismatch between preds and targets"

    diff = preds - targets                          # (N, 2)
    abs_diff = np.abs(diff)

    mae_x = float(abs_diff[:, 0].mean())
    mae_y = float(abs_diff[:, 1].mean())
    mae_overall = float(abs_diff.mean())

    sq_diff = diff ** 2
    rmse_x = float(np.sqrt(sq_diff[:, 0].mean()))
    rmse_y = float(np.sqrt(sq_diff[:, 1].mean()))
    rmse_overall = float(np.sqrt(sq_diff.mean()))

    euclidean = np.sqrt(sq_diff.sum(axis=1))        # (N,)
    euclidean_mean = float(euclidean.mean())
    euclidean_std = float(euclidean.std())

    return {
        "mae_x": mae_x,
        "mae_y": mae_y,
        "mae_overall": mae_overall,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_overall": rmse_overall,
        "euclidean_mean": euclidean_mean,
        "euclidean_std": euclidean_std,
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print a metrics dict returned by compute_metrics."""
    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{tag}Evaluation Metrics")
    print("─" * 40)
    print(f"  MAE   x={metrics['mae_x']:.4f}  y={metrics['mae_y']:.4f}  overall={metrics['mae_overall']:.4f}")
    print(f"  RMSE  x={metrics['rmse_x']:.4f}  y={metrics['rmse_y']:.4f}  overall={metrics['rmse_overall']:.4f}")
    print(f"  Euclidean dist  mean={metrics['euclidean_mean']:.4f}  std={metrics['euclidean_std']:.4f}")
    print("─" * 40)


# ---------------------------------------------------------------------------
# Training history visualization
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses: list, val_losses: list, save_path: str | None = None):
    """Plot train vs. validation loss over epochs."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] Training curves saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Robust loss function
# ---------------------------------------------------------------------------

import torch.nn as nn
import torch

class WeightedHuberLoss(nn.Module):
    """
    Huber loss applied independently to x and y with a higher penalty on x.

    Why Huber over MSE:
        MSE squares large errors — a single bad outlier dominates the gradient.
        Huber is quadratic below `delta` (precise) and linear above (robust).

    Why x_weight=2.0:
        Observed x-MAE is ~2× y-MAE. Up-weighting x forces the optimiser to
        close that gap without affecting y learning.

    Args:
        x_weight : multiplier applied to the x-coordinate Huber loss
        delta    : transition point (in normalised coords). 0.05 ≈ 2 px on 40-px image.
    """

    def __init__(self, x_weight: float = 2.0, delta: float = 0.05):
        super().__init__()
        self.x_weight = x_weight
        self.huber = nn.HuberLoss(reduction="mean", delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_x = self.huber(pred[:, 0], target[:, 0])
        loss_y = self.huber(pred[:, 1], target[:, 1])
        return self.x_weight * loss_x + loss_y


# ---------------------------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------------------------

def find_worst_predictions(preds: np.ndarray,
                           targets: np.ndarray,
                           n: int = 10) -> np.ndarray:
    """
    Return indices of the n samples with the highest Euclidean error (worst first).
    Use these indices against test_loader.dataset.paired_data to inspect images.
    """
    euclidean = np.sqrt(((preds - targets) ** 2).sum(axis=1))
    return np.argsort(euclidean)[::-1][:n]
