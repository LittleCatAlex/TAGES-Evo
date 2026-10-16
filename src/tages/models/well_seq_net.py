"""多尺度一維卷積雙向長短期記憶條件隨機場神經網路 (Multi-Scale 1D-ResNet + BiLSTM + CRF，簡稱 WellboreSeqNet)。"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tages.models.crf import LinearChainCRF

class SqueezeExcitation1D(nn.Module):
    """一維通道注意力機制 (Squeeze-and-Excitation Block)。"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = torch.mean(x, dim=2)  # (B, C)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(2)  # (B, C, 1)
        return x * w


class MultiScaleConv1DBlock(nn.Module):
    """多尺度一維殘差卷積層：以 3m, 5m, 9m 感受野捕捉不同厚度沉積相與構造界面。"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c_branch = out_channels // 3
        c_last = out_channels - c_branch * 2

        self.conv3 = nn.Conv1d(in_channels, c_branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, c_branch, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(in_channels, c_last, kernel_size=9, padding=4)

        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        residual = self.proj(x)
        b3 = self.conv3(x)
        b5 = self.conv5(x)
        b9 = self.conv9(x)
        out = torch.cat([b3, b5, b9], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        out = self.se(out)
        return out + residual


class WellboreSeqNet(nn.Module):
    """井身時序深度神經網路主幹模型。"""

    def __init__(
        self,
        in_features: int = 14,
        cnn_channels: int = 96,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_classes_l2: int = 6,
        n_classes_l1: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes_l2 = n_classes_l2
        self.n_classes_l1 = n_classes_l1

        # 1. 多尺度一維卷積特徵提取
        self.conv_in = nn.Conv1d(in_features, cnn_channels, kernel_size=1)
        self.multiscale_block1 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)
        self.multiscale_block2 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)

        # 2. 雙向長短期記憶網路 (BiLSTM)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        bilstm_out_dim = lstm_hidden * 2  # 256

        # 3. 階層式粗分類頭 (L1 Head)
        self.head_l1 = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes_l1)
        )

        # 4. 融合 L1 特徵之細分類發射頭 (L2 Head)
        self.head_l2 = nn.Sequential(
            nn.Linear(bilstm_out_dim + n_classes_l1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes_l2)
        )

        # 5. 線性鏈條件隨機場層 (CRF)
        self.crf = LinearChainCRF(n_classes_l2)

    def extract_emissions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向提取 L1 Logits 與 L2 CRF 發射分數。
        
        參數：
            x: (B, T, 14)
        回傳：
            emissions_l2: (B, T, n_classes_l2)
            logits_l1: (B, T, n_classes_l1)
        """
        # x: (B, T, C) -> (B, C, T)
        x_cnn = x.transpose(1, 2)
        h = F.relu(self.conv_in(x_cnn))
        h = self.multiscale_block1(h)
        h = self.multiscale_block2(h)
        
        # (B, C, T) -> (B, T, C)
        h_seq = h.transpose(1, 2)
        h_seq = self.dropout(h_seq)

        lstm_out, _ = self.bilstm(h_seq)  # (B, T, 256)

        # L1 粗分類 Logits
        logits_l1 = self.head_l1(lstm_out)  # (B, T, K_L1)

        # 拼接 L1 資訊進一步引導 L2 細分類
        fusion = torch.cat([lstm_out, F.softmax(logits_l1, dim=-1)], dim=-1)
        emissions_l2 = self.head_l2(fusion)  # (B, T, K_L2)

        return emissions_l2, logits_l1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        emissions_l2, logits_l1 = self.extract_emissions(x)
        return {
            "emissions_l2": emissions_l2,
            "logits_l1": logits_l1
        }

    def decode_l2(self, emissions_l2: torch.Tensor, mask: torch.Tensor | None = None) -> List[List[int]]:
        """使用 CRF 維特比演算法解碼最優路徑。"""
        return self.crf.decode(emissions_l2, mask=mask)

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> Dict[str, Any]:
        """推論介面：回傳 L2 類別序列、L1 類別、L2 各層機率與信心度。"""
        self.eval()
        with torch.no_grad():
            emissions_l2, logits_l1 = self.extract_emissions(x)
            
            # CRF 維特比最優路徑
            best_paths_l2 = self.crf.decode(emissions_l2, mask=mask)
            
            # L1 粗分類
            preds_l1 = torch.argmax(logits_l1, dim=-1).cpu().numpy()
            
            # L2 發射邊際機率與信心度
            probs_l2 = F.softmax(emissions_l2, dim=-1).cpu().numpy()
            conf_l2 = np.max(probs_l2, axis=-1)

        return {
            "l2_paths": best_paths_l2,
            "l1_preds": preds_l1,
            "l2_probs": probs_l2,
            "l2_conf": conf_l2
        }
