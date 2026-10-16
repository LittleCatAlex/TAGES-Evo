"""階層式焦點損失與條件隨機場複合損失模組 (Hierarchical Focal-CRF Sequence Loss)。"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedFocalLoss(nn.Module):
    """支援遮罩 (Masked) 的多類別 Focal Loss，專注攻克難分類樣本與不平衡沉積相。"""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        參數：
            logits: (B, T, C)
            targets: (B, T)
            mask: (B, T) BoolTensor
        """
        B, T, C = logits.shape
        flat_logits = logits.view(-1, C)
        flat_targets = targets.view(-1)
        
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # (B*T,)
        pt = torch.exp(-ce_loss)  # 預測該正確類別的機率
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[flat_targets]
            focal_loss = alpha_t * focal_loss

        if mask is not None:
            flat_mask = mask.view(-1).float()
            return torch.sum(focal_loss * flat_mask) / torch.clamp(flat_mask.sum(), min=1.0)
        else:
            return torch.mean(focal_loss)


class HierarchicalSequenceLoss(nn.Module):
    """階層式序列複合損失：聯合優化 CRF 序列轉移、L2 細分類 Focal 損失與 L1 粗分類引導損失。"""

    def __init__(
        self,
        weight_crf: float = 1.0,
        weight_focal_l2: float = 0.5,
        weight_focal_l1: float = 0.3,
        gamma: float = 2.0
    ):
        super().__init__()
        self.weight_crf = weight_crf
        self.weight_focal_l2 = weight_focal_l2
        self.weight_focal_l1 = weight_focal_l1

        self.focal_l2 = MaskedFocalLoss(gamma=gamma)
        self.focal_l1 = MaskedFocalLoss(gamma=gamma)

    def forward(
        self,
        model: nn.Module,
        emissions_l2: torch.Tensor,
        logits_l1: torch.Tensor,
        y_l2: torch.Tensor,
        y_l1: torch.Tensor,
        mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # 1. CRF 序列全域負對數似然
        loss_crf = model.crf.forward_loss(emissions_l2, y_l2, mask=mask)
        # 2. L2 細分類局部 Focal 損失
        loss_focal_l2 = self.focal_l2(emissions_l2, y_l2, mask=mask)
        # 3. L1 粗分類引導損失
        loss_focal_l1 = self.focal_l1(logits_l1, y_l1, mask=mask)

        total_loss = (
            self.weight_crf * loss_crf +
            self.weight_focal_l2 * loss_focal_l2 +
            self.weight_focal_l1 * loss_focal_l1
        )

        return {
            "total_loss": total_loss,
            "loss_crf": loss_crf,
            "loss_focal_l2": loss_focal_l2,
            "loss_focal_l1": loss_focal_l1
        }
