"""
Enhanced Multi-Output DNN with Attention & Ensemble Support
=============================================================
Deeper architecture with self-attention mechanism for cross-target
learning and multi-seed ensemble for robust predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm, SiLU, and Dropout."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.projection = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = self.projection(x)
        out = self.block(x)
        return self.activation(out + residual)


class SelfAttentionBlock(nn.Module):
    """Self-attention to learn cross-feature interactions."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, return_attention=False):
        # x: (batch, dim) → treat as (batch, 1, dim) for attention
        B, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, D)
        
        # PUBLICATION UPGRADE: XAI (Explainable AI) Attention extraction
        if return_attention:
            return x + self.proj(out), attn
        return x + self.proj(out)

class TaskHead(nn.Module):
    """Enhanced task-specific output head."""
    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.head(x)


class MultiOutputRegressor(nn.Module):
    """
    Enhanced multi-output DNN with deeper backbone and attention.

    Architecture:
        Input → SharedBackbone(512→512→256→256→128) → Attention → 5× TaskHead(128→64→32→1)
    """
    def __init__(self, n_features: int, n_targets: int = 5, dropout: float = 0.15):
        super().__init__()
        self.n_targets = n_targets

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Deeper shared backbone
        self.backbone = nn.Sequential(
            ResidualBlock(512, 512, dropout),
            ResidualBlock(512, 256, dropout),
            ResidualBlock(256, 256, dropout),
            ResidualBlock(256, 128, dropout),
        )

        # Attention for feature interactions
        self.attention = SelfAttentionBlock(128, num_heads=4)

        # Deeper task-specific heads
        self.task_heads = nn.ModuleList([
            TaskHead(128, hidden=128, dropout=dropout * 0.7)
            for _ in range(n_targets)
        ])

    def forward(self, x, return_attention=False):
        x = self.input_proj(x)
        shared = self.backbone(x)
        
        if return_attention:
            shared, attn_weights = self.attention(shared, return_attention=True)
        else:
            shared = self.attention(shared, return_attention=False)
            
        outputs = [head(shared) for head in self.task_heads]
        cat_out = torch.cat(outputs, dim=1)
        
        if return_attention:
            return cat_out, attn_weights
        return cat_out

class MaskedMSELoss(nn.Module):
    """MSE Loss ignoring NaN targets, with Sample Weighting support."""
    def forward(self, predictions, targets, mask, sample_weights=None, target_weights=None):
        squared_errors = (predictions - targets) ** 2
        masked_errors = squared_errors * mask
        
        if target_weights is not None:
            masked_errors = masked_errors * target_weights.unsqueeze(0)
            
        if sample_weights is not None:
            masked_errors = masked_errors * sample_weights.unsqueeze(1)
            
        n_valid = mask.sum()
        if n_valid > 0:
            return masked_errors.sum() / n_valid
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)


class PhysicsInformedMaskedLoss(nn.Module):
    """
    Physics-Informed Neural Network (PINN) Loss with three constraints:
      C1: Yield Strength (Idx 1) <= UTS (Idx 2) — yield occurs before fracture
      C2: All predictions >= 0 — no negative mechanical properties
      C3: Tabor relation: HV (Idx 0) ~ k * YS (Idx 1) — hardness-strength proportionality
    """
    def __init__(self, pinn_penalty=10.0, nonneg_penalty=2.0, tabor_penalty=0.1):
        super().__init__()
        self.mse = MaskedMSELoss()
        self.pinn_penalty = pinn_penalty
        self.nonneg_penalty = nonneg_penalty
        self.tabor_penalty = tabor_penalty
        
    def forward(self, predictions, targets, mask, sample_weights=None):
        base_loss = self.mse(predictions, targets, mask, sample_weights)
        
        # C1: Yield Strength (index 1) must be <= UTS (index 2)
        ys_uts_mask = mask[:, 1] * mask[:, 2]
        physical_violation = torch.relu(predictions[:, 1] - predictions[:, 2])
        penalty_c1 = (physical_violation * ys_uts_mask)
        if sample_weights is not None:
            penalty_c1 = penalty_c1 * sample_weights
        n_valid_c1 = ys_uts_mask.sum()
        loss_c1 = (penalty_c1.sum() / max(n_valid_c1, 1)) * self.pinn_penalty
        
        # C2: Non-negativity — all physical properties must be >= 0
        neg_violation = torch.relu(-predictions) * mask  # penalize only valid targets
        loss_c2 = neg_violation.mean() * self.nonneg_penalty
        
        # C3: Tabor relation — HV (idx 0) should correlate with YS (idx 1)
        # In standardized space, enforce soft proportionality: |HV - YS| penalty
        # (both are z-scored, so proportionality becomes approximate equality)
        hv_ys_mask = mask[:, 0] * mask[:, 1]
        tabor_deviation = ((predictions[:, 0] - predictions[:, 1]) ** 2) * hv_ys_mask
        n_valid_c3 = hv_ys_mask.sum()
        loss_c3 = (tabor_deviation.sum() / max(n_valid_c3, 1)) * self.tabor_penalty
        
        return base_loss + loss_c1 + loss_c2 + loss_c3

class MaskedHuberLoss(nn.Module):
    """Huber Loss with NaN masking and Sample Weighting."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions, targets, mask, sample_weights=None, target_weights=None):
        diff = torch.abs(predictions - targets)
        huber = torch.where(diff < self.delta, 0.5 * diff ** 2,
                           self.delta * (diff - 0.5 * self.delta))
        masked = huber * mask
        
        if target_weights is not None:
            masked = masked * target_weights.unsqueeze(0)
            
        if sample_weights is not None:
            masked = masked * sample_weights.unsqueeze(1)
            
        n_valid = mask.sum()
        if n_valid > 0:
            return masked.sum() / n_valid
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MultiOutputRegressor(n_features=70, n_targets=5)
    print(f"Model:\n{model}")
    print(f"\nParameters: {count_parameters(model):,}")
    x = torch.randn(16, 70)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
