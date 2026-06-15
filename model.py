"""
model.py — EfficientNet-B0 backbone with dual linear regression heads.

Architecture:
    EfficientNet-B0 (ImageNet pretrained) → GlobalAvgPool
    → Dropout(0.4) → FC(1280→512) → BN → ReLU
    → Dropout(0.2) → FC(512→256)  → BN → ReLU
    → Head-X: Linear(256→1)   [x pupil center, normalised]
    → Head-Y: Linear(256→1)   [y pupil center, normalised]

Output shape: (batch, 2)  — [x, y] continuous values in [0, 1]
No softmax. No sigmoid. Pure linear regression.

Changes from v1:
  - Neck deepened: 1280→256 (1 layer)  →  1280→512→256 (2 layers)
  - BatchNorm added after each FC: stabilises training, acts as regulariser
  - Default dropout raised 0.3 → 0.4: counters the observed overfitting
  - Head init switched to Xavier (better for the final linear regression layer)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GazeEstimationModel(nn.Module):
    """
    Transfer-learning gaze estimator built on EfficientNet-B0.

    Args:
        pretrained   : load ImageNet weights for the backbone (default True)
        dropout_rate : dropout applied before the shared FC layer
        hidden_dim   : width of the shared FC layer (default 256)
        freeze_backbone : freeze backbone weights for initial warm-up (optional)
    """

    def __init__(self,
                 pretrained: bool = True,
                 dropout_rate: float = 0.4,
                 hidden_dim: int = 256,
                 freeze_backbone: bool = False):
        super().__init__()

        # --- Backbone ---
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Keep only the feature extractor and the adaptive avg pool
        self.features = backbone.features       # conv layers → (B, 1280, H, W)
        self.avgpool  = backbone.avgpool        # → (B, 1280, 1, 1)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # EfficientNet-B0 outputs 1280 channels after avgpool
        feat_dim = 1280

        # --- Shared neck: two FC layers with BatchNorm ---
        # BatchNorm provides per-layer normalisation and acts as a regulariser,
        # reducing the need to rely solely on dropout for overfitting control.
        self.neck = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),   # lighter dropout in second layer
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Regression heads (one per coordinate, pure linear) ---
        self.head_x = nn.Linear(hidden_dim, 1)
        self.head_y = nn.Linear(hidden_dim, 1)

        # Weight init for new layers
        self._init_new_layers()

    def _init_new_layers(self):
        for module in self.neck:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        # Xavier for the final regression output (no ReLU follows)
        for head in [self.head_x, self.head_y]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224)
        Returns:
            coords : (B, 2)  — [x_pred, y_pred] in [0, 1]
        """
        x = self.features(x)               # (B, 1280, 7, 7)
        x = self.avgpool(x)                # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)           # (B, 1280)
        x = self.neck(x)                   # (B, hidden_dim)

        x_pred = self.head_x(x)           # (B, 1)
        y_pred = self.head_y(x)           # (B, 1)

        # clamp keeps predictions inside the valid image coordinate range [0, 1]
        return torch.cat([x_pred, y_pred], dim=1).clamp(0.0, 1.0)   # (B, 2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-dim backbone features (no neck/head). Used by PersonalizedGazeMapper."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)   # (B, 1280)

    def forward_with_features(self, x: torch.Tensor):
        """Single forward pass — returns (coords, features) to avoid computing backbone twice."""
        feat   = self.extract_features(x)                              # (B, 1280)
        neck   = self.neck(feat)                                       # (B, 256)
        coords = torch.cat([self.head_x(neck), self.head_y(neck)], dim=1).clamp(0.0, 1.0)
        return coords, feat                                            # (B, 2), (B, 1280)

    def unfreeze_backbone(self):
        """Call after initial warm-up to fine-tune the entire network."""
        for p in self.features.parameters():
            p.requires_grad = True
        print("[Model] Backbone unfrozen for full fine-tuning.")

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = GazeEstimationModel(pretrained=False)   # skip download in quick test
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"
    stats = model.count_parameters()
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {stats['total']:,} total / {stats['trainable']:,} trainable")
