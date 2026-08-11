"""The multimodal task-demand decoder.

One small temporal encoder per physiological modality, concatenated and passed
through a two-layer MLP. Deliberately shallow: the epochs are 2 s long and there
are only a few thousand per subject.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset


def init_weights_kaiming(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SimpleTemporalEncoder(nn.Module):
    """Two Conv1d+BatchNorm+ReLU blocks, then global average pooling over time.

    Input (B, C_in, T) -> output (B, d_model).
    """

    def __init__(self, in_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)


class ModalityEncodersSimple(nn.Module):
    """One ``SimpleTemporalEncoder`` per entry of ``modalities``."""

    def __init__(self, modalities: dict, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.modalities = list(modalities.keys())
        self.groups = modalities
        self.encoders = nn.ModuleDict({
            name: SimpleTemporalEncoder(len(idx), d_model, dropout)
            for name, idx in modalities.items()
        })
        self.out_dim = d_model * len(self.modalities)

    def forward(self, x):
        feats = [self.encoders[m](x[:, self.groups[m], :]) for m in self.modalities]
        return torch.cat(feats, dim=-1)


class MultiModalClassifierSimple(nn.Module):
    """Per-modality encoders -> concatenation -> MLP -> 2 logits."""

    def __init__(self, modalities: dict, d_model: int = 64, mlp_hidden: int = 128,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = ModalityEncodersSimple(modalities, d_model, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )
        self.apply(init_weights_kaiming)

    def forward(self, x, return_features: bool = False):
        z = self.backbone(x)
        logits = self.mlp(z)
        if return_features:
            return {"logits": logits, "features": z}
        return {"logits": logits}


class LogitsOnly(nn.Module):
    """Adapter exposing a plain-tensor forward pass, as Captum expects."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)["logits"]


class MultiModalDataset(Dataset):
    """Epoch dataset with the two augmentations used during training.

    Temporal masking and additive Gaussian noise are each applied with p = 0.5,
    independently of one another.
    """

    def __init__(self, X, y, augment=True, mask_prob=0.15, noise_std=0.1):
        self.X, self.y = X, y
        self.augment = augment
        self.mask_prob = mask_prob
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            if torch.rand(1) < 0.5:
                mask_len = int(x.size(1) * self.mask_prob)
                start = torch.randint(0, x.size(1) - mask_len, (1,))
                x[:, start:start + mask_len] = 0
            if torch.rand(1) < 0.5:
                x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx]
