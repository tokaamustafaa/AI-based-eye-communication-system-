"""
train.py — Full training pipeline.

Usage (from your notebook):

    from dataset import get_dataloaders
    from model   import GazeEstimationModel
    from train   import train

    train_loader, val_loader, _ = get_dataloaders(all_paired_data, batch_size=8)
    model = GazeEstimationModel(pretrained=True)
    train(model, train_loader, val_loader)

Changes from v1:
  - Loss: MSELoss → WeightedHuberLoss (robust to outliers; 2x penalty on x)
  - Gradient accumulation: effective batch = batch_size × grad_accum_steps
    (default: 8 × 4 = 32) — stable gradients without extra RAM
  - weight_decay raised: 1e-4 → 3e-4 (stronger L2 against overfitting)
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import EarlyStopping, WeightedHuberLoss, save_checkpoint, plot_training_curves


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer | None,
               device: torch.device,
               is_train: bool,
               grad_accum_steps: int = 1) -> float:
    """
    Run one full pass over `loader`.

    Gradient accumulation (is_train only):
        Instead of stepping the optimiser every batch, gradients are accumulated
        for `grad_accum_steps` batches before a single optimiser step.
        Effective batch size = batch_size × grad_accum_steps.
        This gives more stable gradient estimates without extra memory.

    Returns average loss over the epoch.
    """
    model.train(is_train)
    total_loss = 0.0
    n_samples  = 0

    if is_train:
        assert optimizer is not None, "optimizer required for training"
        optimizer.zero_grad()

    with torch.set_grad_enabled(is_train):
        for step, (imgs, labels) in enumerate(loader, start=1):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds = model(imgs)                         # (B, 2)
            loss  = criterion(preds, labels)

            if is_train:
                assert optimizer is not None
                # Divide by accum steps so the effective gradient magnitude
                # matches a single large-batch update
                (loss / grad_accum_steps).backward()

                if step % grad_accum_steps == 0 or step == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * imgs.size(0)
            n_samples  += imgs.size(0)

    return total_loss / n_samples


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          *,
          num_epochs: int = 100,
          learning_rate: float = 5e-4,
          weight_decay: float = 3e-4,
          grad_accum_steps: int = 4,
          early_stopping_patience: int = 20,
          scheduler_patience: int = 6,
          scheduler_factor: float = 0.5,
          x_loss_weight: float = 2.0,
          huber_delta: float = 0.05,
          checkpoint_path: str = "checkpoints/best_model.pth",
          device: torch.device | None = None,
          plot_curves: bool = True) -> dict:
    """
    Full training loop with WeightedHuberLoss, AdamW, ReduceLROnPlateau,
    gradient accumulation, and early stopping.

    Args:
        model                   : GazeEstimationModel instance
        train_loader / val_loader: from get_dataloaders()
        num_epochs              : max epochs
        learning_rate           : initial LR (lowered to 5e-4 for stability)
        weight_decay            : L2 regularisation (raised to 3e-4)
        grad_accum_steps        : batches between optimiser steps; effective
                                  batch = batch_size × grad_accum_steps
        early_stopping_patience : epochs without val improvement before stop
        scheduler_patience      : epochs without improvement before LR drop
        scheduler_factor        : LR reduction factor on plateau
        x_loss_weight           : extra penalty on x-coordinate errors
        huber_delta             : Huber transition point (normalised coords)
        checkpoint_path         : path to save best weights
        device                  : auto-detected when None
        plot_curves             : show loss curves at end

    Returns:
        history dict: {"train_loss": [...], "val_loss": [...]}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device          : {device}")
    batch_size = train_loader.batch_size or 1
    print(f"[Train] Effective batch : {batch_size} × {grad_accum_steps} = "
          f"{batch_size * grad_accum_steps}")
    model = model.to(device)

    criterion = WeightedHuberLoss(x_weight=x_loss_weight, delta=huber_delta)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    print(f"[Train] Starting — max_epochs={num_epochs}, lr={learning_rate}, "
          f"x_weight={x_loss_weight}, huber_delta={huber_delta}")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = _run_epoch(model, train_loader, criterion, optimizer,
                                device, is_train=True,
                                grad_accum_steps=grad_accum_steps)
        val_loss   = _run_epoch(model, val_loader,   criterion, None,
                                device, is_train=False)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed  = time.time() - t0
        prev_lr  = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        cur_lr   = optimizer.param_groups[0]["lr"]
        lr_tag   = f"  ↓ LR→{cur_lr:.2e}" if cur_lr < prev_lr else ""

        print(f"Epoch [{epoch:03d}/{num_epochs}]  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={cur_lr:.2e}  ({elapsed:.1f}s){lr_tag}")

        if val_loss <= early_stopping.best_loss:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        if early_stopping.step(val_loss, model):
            break

    early_stopping.restore_best_weights(model)

    print("=" * 65)
    print(f"[Train] Finished. Best val_loss = {early_stopping.best_loss:.6f}")

    if plot_curves:
        curves_path = os.path.join(
            os.path.dirname(checkpoint_path) or ".", "training_curves.png"
        )
        plot_training_curves(history["train_loss"], history["val_loss"],
                             save_path=curves_path)

    return history


# ---------------------------------------------------------------------------
# Optional: two-stage warm-up then fine-tune
# ---------------------------------------------------------------------------

def train_two_stage(model: nn.Module,
                    train_loader,
                    val_loader,
                    *,
                    warmup_epochs: int = 5,
                    warmup_lr: float = 1e-3,
                    finetune_lr: float = 2e-4,
                    **train_kwargs) -> dict:
    """
    Stage 1: Train only neck + heads (backbone frozen).
    Stage 2: Unfreeze backbone and fine-tune everything.
    """
    print("[Train] === Stage 1: Warm-up (backbone frozen) ===")
    ckpt_path = train_kwargs.pop("checkpoint_path", "checkpoints/best_model.pth")
    history_warmup = train(
        model, train_loader, val_loader,
        num_epochs=warmup_epochs,
        learning_rate=warmup_lr,
        checkpoint_path=ckpt_path,
        plot_curves=False,
        **train_kwargs,
    )

    model.unfreeze_backbone()

    print("\n[Train] === Stage 2: Full fine-tuning ===")
    history_finetune = train(
        model, train_loader, val_loader,
        learning_rate=finetune_lr,
        checkpoint_path=ckpt_path,
        **train_kwargs,
    )

    return {
        "train_loss": history_warmup["train_loss"] + history_finetune["train_loss"],
        "val_loss":   history_warmup["val_loss"]   + history_finetune["val_loss"],
    }
