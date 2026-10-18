"""階層式焦點損失、條件隨機場與岩石物理引導複合損失模組 (Hierarchical Physics-Guided Sequence Loss)。

本模組落實 Phase B 之步驟 B1：
1. 支援遮罩 (Masked) 的多類別 Focal Loss，結合類別平衡反比平方根權重 (Class-Balanced Weights)。
2. 條件隨機場 (Linear-Chain CRF) 序列轉移全域負對數似然損失 (Forward NLL Loss)。
3. 岩石物理一致性懲罰項 (Rock-Physics Consistency Penalty L_phy)：
   - 淺層極低剪力波速 (Vs < 200 m/s) 誤判純礫石懲罰
   - 高泊松比 (nu > 0.42) 水飽和泥質區誤判粗顆粒骨架相懲罰
   - 深部高波速區 (Vs > 450 m/s) 誤判鬆散軟泥懲罰
   - 極淺層表土軟弱區粗粒倒置懲罰
4. 複合目標函數：
   L_total = L_CRF + lambda_1 * L_Focal-L2 + lambda_2 * L_Focal-L1 + lambda_3 * L_phy
"""
from __future__ import annotations
import numpy as np
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


class RockPhysicsConsistencyLoss(nn.Module):
    """岩石物理一致性懲罰項 (Rock-Physics Consistency Loss, L_phy)。
    
    依據沉積物岩石物理學原理，對違反速度-孔隙-泊松比物理先驗邊界的預測施加平滑懲罰：
    1. 低波速粗粒懲罰：極低剪力波速 (Vs < 200 m/s) 誤判為堅硬粗礫石。
    2. 高泊松比泥質懲罰：水飽和細粒高泊松比 (nu > 0.42) 誤判為粗骨架粗砂/礫石。
    3. 深部高波速軟泥懲罰：高剪力波速 (Vs > 450 m/s) 誤判為未固結鬆散泥。
    4. 淺層波速倒置懲罰：極淺部未壓密區 (z < 25m, Vs < 180 m/s) 誤判粗粒相。
    """

    def __init__(
        self,
        scaler_mean: np.ndarray | torch.Tensor | None = None,
        scaler_scale: np.ndarray | torch.Tensor | None = None,
        l2_classes: list[int] | None = None,
        thresholds: dict | None = None
    ):
        super().__init__()
        self.l2_classes = l2_classes or [2, 3, 4, 5, 6, 7]
        # 類別對應索引：2:Clay(0), 3:Silt(1), 4:FineSand(2), 5:CoarseSand(3), 6:Mixed(4), 7:Gravel(5)
        self.idx_clay = self.l2_classes.index(2) if 2 in self.l2_classes else 0
        self.idx_coarsesand = self.l2_classes.index(5) if 5 in self.l2_classes else min(3, len(self.l2_classes)-1)
        self.idx_gravel = self.l2_classes.index(7) if 7 in self.l2_classes else min(5, len(self.l2_classes)-1)

        self.thresholds = thresholds or {}

        # 註冊標準化反變換係數 (若未提供則使用典型全島觀測井平均與標準差)
        # FEATURE_NAMES 索引：depth_m: 0, vs_ms: 4, poisson_ratio: 6
        if scaler_mean is not None and scaler_scale is not None:
            m = torch.tensor(scaler_mean, dtype=torch.float32)
            s = torch.tensor(scaler_scale, dtype=torch.float32)
            self.register_buffer("mean_depth", m[0])
            self.register_buffer("scale_depth", s[0])
            self.register_buffer("mean_vs", m[4])
            self.register_buffer("scale_vs", s[4])
            self.register_buffer("mean_poisson", m[6])
            self.register_buffer("scale_poisson", s[6])
        else:
            self.register_buffer("mean_depth", torch.tensor(45.0))
            self.register_buffer("scale_depth", torch.tensor(35.0))
            self.register_buffer("mean_vs", torch.tensor(280.0))
            self.register_buffer("scale_vs", torch.tensor(90.0))
            self.register_buffer("mean_poisson", torch.tensor(0.32))
            self.register_buffer("scale_poisson", torch.tensor(0.06))

        # 根據資料集物理尺度自適應門檻 (地體層析尺度 Vs > 1000 m/s vs. 淺層工程測井尺度 Vs < 600 m/s)
        is_crustal_scale = float(self.mean_vs) > 1000.0
        self.vs_gravel_min = self.thresholds.get("vs_gravel_min", 1650.0 if is_crustal_scale else 200.0)
        self.vs_gravel_scale = 100.0 if is_crustal_scale else 50.0
        self.poisson_gravel_max = self.thresholds.get("poisson_gravel_max", 0.330 if is_crustal_scale else 0.40)
        self.poisson_scale = 0.03 if is_crustal_scale else 0.05
        self.vs_clay_max = self.thresholds.get("vs_clay_max", 2500.0 if is_crustal_scale else 450.0)
        self.vs_clay_scale = 100.0
        self.shallow_depth = self.thresholds.get("shallow_depth", 25.0)
        self.shallow_vs_min = self.thresholds.get("shallow_vs_min", 1650.0 if is_crustal_scale else 180.0)

    def forward(
        self,
        emissions: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        參數：
            emissions: (B, T, C) 未歸一化發射分數
            x: (B, T, D) 標準化特徵矩陣
            mask: (B, T) BoolTensor 有效深度遮罩
        """
        probs = F.softmax(emissions, dim=-1) # (B, T, C)
        p_clay = probs[..., self.idx_clay]
        p_coarse = probs[..., self.idx_coarsesand]
        p_gravel = probs[..., self.idx_gravel]

        # 1. 物理量反歸一化 (還原為真實物理量)
        depth = x[..., 0] * self.scale_depth + self.mean_depth       # (m)
        vs = x[..., 4] * self.scale_vs + self.mean_vs                # (m/s)
        poisson = x[..., 6] * self.scale_poisson + self.mean_poisson # (無因次)

        # 2. 懲罰項計算 (平滑可微分 ReLU 懲罰)
        # (a) 極低波速誤判為純礫石
        p1 = F.relu((self.vs_gravel_min - vs) / self.vs_gravel_scale) * p_gravel

        # (b) 高泊松比水飽和泥質區誤判為粗顆粒骨架
        p2 = F.relu((poisson - self.poisson_gravel_max) / self.poisson_scale) * (p_gravel + 0.6 * p_coarse)

        # (c) 深部超高波速區誤判為未壓密軟泥
        p3 = F.relu((vs - self.vs_clay_max) / self.vs_clay_scale) * p_clay

        # (d) 淺部低波速粗顆粒倒置懲罰
        is_shallow = (depth < self.shallow_depth).float()
        p4 = F.relu((self.shallow_vs_min - vs) / self.vs_gravel_scale) * (p_coarse + p_gravel) * is_shallow

        penalty = p1 + p2 + p3 + p4

        if mask is not None:
            m_flat = mask.float()
            return torch.sum(penalty * m_flat) / torch.clamp(m_flat.sum(), min=1.0)
        else:
            return torch.mean(penalty)


class HierarchicalSequenceLoss(nn.Module):
    """階層式序列物理引導複合損失：
    L_total = L_CRF + lambda_1 * L_Focal-L2 + lambda_2 * L_Focal-L1 + lambda_3 * L_phy
    """

    def __init__(
        self,
        weight_crf: float = 1.0,
        weight_focal_l2: float = 0.5,
        weight_focal_l1: float = 0.3,
        weight_phy: float = 0.2,
        gamma: float = 2.0,
        alpha_l2: torch.Tensor | None = None,
        phy_loss_fn: RockPhysicsConsistencyLoss | None = None
    ):
        super().__init__()
        self.weight_crf = weight_crf
        self.weight_focal_l2 = weight_focal_l2
        self.weight_focal_l1 = weight_focal_l1
        self.weight_phy = weight_phy

        self.focal_l2 = MaskedFocalLoss(gamma=gamma, alpha=alpha_l2)
        self.focal_l1 = MaskedFocalLoss(gamma=gamma)
        self.phy_loss_fn = phy_loss_fn or RockPhysicsConsistencyLoss()

    def forward(
        self,
        model: nn.Module,
        emissions_l2: torch.Tensor,
        logits_l1: torch.Tensor,
        y_l2: torch.Tensor,
        y_l1: torch.Tensor,
        mask: torch.Tensor,
        x: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        # 1. CRF 序列全域負對數似然
        loss_crf = model.crf.forward_loss(emissions_l2, y_l2, mask=mask)
        # 2. L2 細分類局部 Focal 損失
        loss_focal_l2 = self.focal_l2(emissions_l2, y_l2, mask=mask)
        # 3. L1 粗分類引導損失
        loss_focal_l1 = self.focal_l1(logits_l1, y_l1, mask=mask)
        # 4. 岩石物理一致性懲罰項 L_phy
        if x is not None and self.weight_phy > 0:
            loss_phy = self.phy_loss_fn(emissions_l2, x, mask=mask)
        else:
            loss_phy = torch.tensor(0.0, device=emissions_l2.device)

        total_loss = (
            self.weight_crf * loss_crf +
            self.weight_focal_l2 * loss_focal_l2 +
            self.weight_focal_l1 * loss_focal_l1 +
            self.weight_phy * loss_phy
        )

        return {
            "total_loss": total_loss,
            "loss_crf": loss_crf,
            "loss_focal_l2": loss_focal_l2,
            "loss_focal_l1": loss_focal_l1,
            "loss_phy": loss_phy
        }
