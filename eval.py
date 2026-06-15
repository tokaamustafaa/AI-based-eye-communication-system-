"""
eval.py — Evaluation pipeline: inference on test set, metrics, and visualizations.

Usage (from your notebook / script):

    from dataset import get_dataloaders
    from model   import GazeEstimationModel
    from eval    import evaluate, visualize_predictions

    _, _, test_loader = get_dataloaders(all_paired_data)
    model = GazeEstimationModel()
    evaluate(model, test_loader, checkpoint_path="checkpoints/best_model.pth")
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import load_checkpoint, compute_metrics, print_metrics


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

def _collect_predictions(model: torch.nn.Module,
                          loader: DataLoader,
                          device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on every batch in `loader`.
    Returns:
        preds   : (N, 2) float32 numpy array
        targets : (N, 2) float32 numpy array
    """
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs).cpu().numpy()           # (B, 2)
            all_preds.append(preds)
            all_targets.append(labels.numpy())

    return np.vstack(all_preds), np.vstack(all_targets)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module,
             test_loader: DataLoader,
             *,
             checkpoint_path: str | None = None,
             device: torch.device | None = None,
             verbose: bool = True) -> dict:
    """
    Load checkpoint (optional), run inference on test_loader, compute metrics.

    Args:
        model            : GazeEstimationModel instance (architecture must match checkpoint)
        test_loader      : DataLoader for the test split
        checkpoint_path  : path to saved .pth checkpoint; skipped when None
        device           : torch.device; auto-detected when None
        verbose          : print metrics to stdout

    Returns:
        metrics dict (see utils.compute_metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    model = model.to(device)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=device)

    preds, targets = _collect_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, targets)

    if verbose:
        print_metrics(metrics, prefix="Test")

    return metrics


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def visualize_predictions(model: torch.nn.Module,
                           test_loader: DataLoader,
                           *,
                           checkpoint_path: str | None = None,
                           device: torch.device | None = None,
                           n_samples: int = 16,
                           save_path: str | None = None):
    """
    Scatter plot: predicted vs. ground-truth pupil centers.
    A perfect model would have all points on the diagonal.

    Args:
        n_samples : how many samples to show (subsample if dataset is large)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=device)

    preds, targets = _collect_predictions(model, test_loader, device)

    # Subsample for readability
    idx = np.random.choice(len(preds), size=min(n_samples, len(preds)), replace=False)
    preds_s   = preds[idx]
    targets_s = targets[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dim, label in zip(axes, [0, 1], ["X (horizontal)", "Y (vertical)"]):
        ax.scatter(targets_s[:, dim], preds_s[:, dim], alpha=0.7, edgecolors="k", linewidths=0.5)
        # Perfect prediction diagonal
        lo = min(targets_s[:, dim].min(), preds_s[:, dim].min()) - 0.05
        hi = max(targets_s[:, dim].max(), preds_s[:, dim].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Pupil Center — {label}")
        ax.legend()
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

    plt.suptitle("Prediction vs. Ground Truth", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] Scatter saved → {save_path}")
    plt.show()


def visualize_error_distribution(model: torch.nn.Module,
                                  test_loader: DataLoader,
                                  *,
                                  checkpoint_path: str | None = None,
                                  device: torch.device | None = None,
                                  save_path: str | None = None):
    """
    Histogram of per-sample Euclidean errors + X/Y error distributions.
    Helps identify tail-risk samples and systematic bias.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=device)

    preds, targets = _collect_predictions(model, test_loader, device)

    diff       = preds - targets                        # (N, 2)
    euclidean  = np.sqrt((diff ** 2).sum(axis=1))      # (N,)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Euclidean distance histogram
    axes[0].hist(euclidean, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(euclidean.mean(), color="red", linestyle="--",
                    label=f"Mean={euclidean.mean():.4f}")
    axes[0].set_xlabel("Euclidean Error (normalized)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Euclidean Distance Error Distribution")
    axes[0].legend()

    # X error histogram
    axes[1].hist(diff[:, 0], bins=30, color="darkorange", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].axvline(diff[:, 0].mean(), color="red", linestyle="--",
                    label=f"Bias={diff[:, 0].mean():.4f}")
    axes[1].set_xlabel("Δx  (pred − gt)")
    axes[1].set_title("X Error Distribution")
    axes[1].legend()

    # Y error histogram
    axes[2].hist(diff[:, 1], bins=30, color="seagreen", edgecolor="white", linewidth=0.5)
    axes[2].axvline(0, color="black", linestyle="-", linewidth=0.8)
    axes[2].axvline(diff[:, 1].mean(), color="red", linestyle="--",
                    label=f"Bias={diff[:, 1].mean():.4f}")
    axes[2].set_xlabel("Δy  (pred − gt)")
    axes[2].set_title("Y Error Distribution")
    axes[2].legend()

    plt.suptitle("Error Analysis", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] Error distribution saved → {save_path}")
    plt.show()


def visualize_spatial_errors(model: torch.nn.Module,
                              test_loader: DataLoader,
                              *,
                              checkpoint_path: str | None = None,
                              device: torch.device | None = None,
                              save_path: str | None = None):
    """
    2D error map: plot ground-truth positions color-coded by Euclidean error.
    Reveals spatial regions where the model struggles (e.g. extreme gaze angles).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, device=device)

    preds, targets = _collect_predictions(model, test_loader, device)
    euclidean = np.sqrt(((preds - targets) ** 2).sum(axis=1))

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        targets[:, 0], targets[:, 1],
        c=euclidean, cmap="RdYlGn_r",
        s=20, alpha=0.7, edgecolors="none"
    )
    plt.colorbar(sc, label="Euclidean Error (normalized)")
    plt.xlabel("Ground-truth X")
    plt.ylabel("Ground-truth Y")
    plt.title("Spatial Error Map\n(green=low error, red=high error)")
    plt.gca().invert_yaxis()    # match image coordinate convention (y down)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [Plot] Spatial error map saved → {save_path}")
    plt.show()
