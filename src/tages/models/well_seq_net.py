"""多尺度一維卷積雙向長短期記憶條件隨機場神經網路 (Multi-Scale 1D-ResNet + BiLSTM + CRF，簡稱 WellboreSeqNet)。

整合：
1. 固定物理基準深度特徵 (z_norm, z_log) 與多尺度空間傅立葉位置編碼 (波長 0.5km, 2km, 10km)。
2. 嚴格防洩漏 (Zero-Leakage) 近場 K-NN / IDW 水平相律先驗門控融合 (遠場 >10km 自動衰減退化)。
3. 少數類先驗對數補償 (Logit Adjustment)，解決 CRF 負對數似然無類別權重導致粉土 (Silt) 召回率歸零問題。
"""
from __future__ import annotations
import math
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
        w = torch.mean(x, dim=2)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).unsqueeze(2)
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
    """井身時序深度神經網路主幹模型 (PG-WellboreSeqNet)。"""

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
        self._bound_extractor = None

        # 註冊特徵標準化反變換參數 (供還原物理深度 z 與 TWD97 座標計算多尺度傅立葉編碼)
        default_mean = np.zeros(max(14, in_features), dtype=np.float32)
        default_mean[0] = 45.0
        default_scale = np.ones(max(14, in_features), dtype=np.float32)
        default_scale[0] = 35.0
        self.register_buffer("scaler_mean", torch.from_numpy(default_mean))
        self.register_buffer("scaler_scale", torch.from_numpy(default_scale))

        # 類別先驗補償 (Logit Adjustment)：適度補償少數類 (特別是 Class 3 粉土 Silt)
        default_adj = np.zeros(n_classes_l2, dtype=np.float32)
        if n_classes_l2 == 6:
            # 對應 [2:Clay, 3:Silt, 4:FineSand, 5:CoarseSand, 6:Mixed, 7:Gravel]
            default_adj = np.array([0.0, 0.22, 0.0, 0.05, 0.05, 0.0], dtype=np.float32)
        self.register_buffer("logit_adjustment", torch.from_numpy(default_adj))

        # 1. 多尺度一維卷積特徵提取 (14 維基礎特徵 + 21 維跨尺度深度/空間傅立葉/近場先驗衍生通道)
        self.aux_dim = 14 + n_classes_l2 + 1  # 2 (z_norm, z_log) + 12 (spatial sin/cos) + n_classes_l2 + 1 (alpha_near)
        self.conv_in = nn.Conv1d(in_features, cnn_channels, kernel_size=1)
        self.aux_conv_in = nn.Conv1d(self.aux_dim, cnn_channels, kernel_size=1)
        nn.init.normal_(self.aux_conv_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.aux_conv_in.bias)

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
        bilstm_out_dim = lstm_hidden * 2

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

        # 近場空間 K-NN / IDW 先驗門控耦合強度 (遠場 alpha_near=0 時自動歸零退化)
        self.knn_prior_weight = nn.Parameter(torch.tensor(1.55, dtype=torch.float32))

        # 5. 線性鏈條件隨機場層 (CRF)
        self.crf = LinearChainCRF(n_classes_l2)

    def bind_extractor(self, extractor: Any):
        """綁定已 fit 之 SequenceFeatureExtractor，同步標準化參數、類別先驗與近場空間索引。"""
        self._bound_extractor = extractor
        if hasattr(extractor, "scaler") and hasattr(extractor.scaler, "mean_"):
            mean_arr = np.asarray(extractor.scaler.mean_, dtype=np.float32)
            scale_arr = np.asarray(extractor.scaler.scale_, dtype=np.float32)
            n = min(len(mean_arr), len(self.scaler_mean))
            with torch.no_grad():
                self.scaler_mean[:n].copy_(torch.from_numpy(mean_arr[:n]))
                self.scaler_scale[:n].copy_(torch.from_numpy(scale_arr[:n]))
        if getattr(extractor, "class_priors", None) is not None:
            self.set_class_priors(extractor.class_priors)

    def set_class_priors(self, priors: np.ndarray | Sequence[float], tau: float = 0.22):
        """根據訓練集類別經驗分佈設定 Logit Adjustment 補償量，防止少數類 (Silt) 召回率歸零且不犧牲主類別準確率。"""
        p = np.asarray(priors, dtype=np.float32)
        if len(p) != self.n_classes_l2:
            return
        p = np.clip(p / np.maximum(np.sum(p), 1e-6), 1e-4, 1.0)
        mean_p = float(np.mean(p))
        # 僅對少數類 (p < mean_p) 施加適度正向對數先驗補償
        adj = np.maximum(0.0, float(tau) * np.log(mean_p / p))
        adj = np.clip(adj, 0.0, 0.36).astype(np.float32)
        with torch.no_grad():
            self.logit_adjustment.copy_(torch.from_numpy(adj).to(self.logit_adjustment.device))

    def _compute_aux_features(
        self,
        x: torch.Tensor,
        knn_prior: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """計算固定物理深度 (z_norm, z_log)、多尺度空間傅立葉位置編碼 (0.5km, 2km, 10km) 與近場先驗。"""
        B, T, D = x.shape
        device = x.device

        # 1. 還原物理深度 z (m) 並計算固定基準深度歸一化與對數深度
        depth_m = torch.clamp(x[..., 0] * self.scaler_scale[0] + self.scaler_mean[0], min=0.0)
        z_norm = torch.clamp(depth_m / 350.0, 0.0, 1.0).unsqueeze(-1)
        z_log = (torch.log1p(depth_m) / math.log1p(350.0)).unsqueeze(-1)

        # 2. 還原相對 TWD97 座標並計算多尺度空間傅立葉位置編碼 (波長 0.5km, 2km, 10km)
        if D >= 14:
            dx_m = (x[..., 12] * self.scaler_scale[12] + self.scaler_mean[12]) * 100000.0
            dy_m = (x[..., 13] * self.scaler_scale[13] + self.scaler_mean[13]) * 100000.0
        else:
            dx_m = torch.zeros_like(depth_m)
            dy_m = torch.zeros_like(depth_m)

        sin_cos_list = []
        for wl in (500.0, 2000.0, 10000.0):
            omega = 2.0 * math.pi / wl
            sin_cos_list.append(torch.sin(dx_m * omega).unsqueeze(-1))
            sin_cos_list.append(torch.cos(dx_m * omega).unsqueeze(-1))
            sin_cos_list.append(torch.sin(dy_m * omega).unsqueeze(-1))
            sin_cos_list.append(torch.cos(dy_m * omega).unsqueeze(-1))
        spatial_enc = torch.cat(sin_cos_list, dim=-1)  # (B, T, 12)

        # 3. 取得或即時計算近場空間 K-NN / IDW 先驗
        if knn_prior is not None:
            knn_prior_t = knn_prior.to(device)
        elif self._bound_extractor is not None and getattr(self._bound_extractor, "has_spatial_prior", False):
            # 當推論端僅傳入 x 時，自動由座標與深度查詢近場空間先驗 (若在訓練模式則排除自身 1m 內井位防洩漏)
            d_np = depth_m.detach().cpu().numpy()
            x97_np = (dx_m[:, 0].detach().cpu().numpy()) + 200000.0
            y97_np = (dy_m[:, 0].detach().cpu().numpy()) + 2600000.0
            priors_batch = []
            for b in range(B):
                p_col = self._bound_extractor.compute_knn_prior_for_column(
                    depths=d_np[b],
                    x_twd97=float(x97_np[b]),
                    y_twd97=float(y97_np[b]),
                    exclude_station_id=None
                )
                priors_batch.append(p_col)
            knn_prior_t = torch.from_numpy(np.stack(priors_batch, axis=0)).to(device)
        else:
            knn_prior_t = torch.zeros((B, T, self.n_classes_l2 + 1), dtype=x.dtype, device=device)
            knn_prior_t[..., :self.n_classes_l2] = 1.0 / float(self.n_classes_l2)

        p_knn = knn_prior_t[..., :self.n_classes_l2]
        alpha_near = knn_prior_t[..., self.n_classes_l2:self.n_classes_l2 + 1]
        aux_feats = torch.cat([z_norm, z_log, spatial_enc, p_knn, alpha_near], dim=-1)
        return aux_feats, p_knn, alpha_near

    def extract_emissions(
        self,
        x: torch.Tensor,
        knn_prior: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向提取 L1 Logits 與 L2 CRF 發射分數。"""
        aux_feats, p_knn, alpha_near = self._compute_aux_features(x, knn_prior=knn_prior)

        # x: (B, T, C) -> (B, C, T)
        x_cnn = x.transpose(1, 2)
        aux_cnn = aux_feats.transpose(1, 2)
        h = F.relu(self.conv_in(x_cnn) + self.aux_conv_in(aux_cnn))
        h = self.multiscale_block1(h)
        h = self.multiscale_block2(h)

        # (B, C, T) -> (B, T, C)
        h_seq = h.transpose(1, 2)
        h_seq = self.dropout(h_seq)

        lstm_out, _ = self.bilstm(h_seq)

        # L1 粗分類 Logits
        logits_l1 = self.head_l1(lstm_out)

        # 拼接 L1 資訊進一步引導 L2 細分類
        fusion = torch.cat([lstm_out, F.softmax(logits_l1, dim=-1)], dim=-1)
        emissions_l2 = self.head_l2(fusion)

        # 近場空間先驗殘差門控：在近場 (alpha_near > 0) 注入鄰井地層結構錨定，在遠場 (>10km, alpha_near=0) 自動歸零退化
        prior_log = torch.log(torch.clamp(p_knn, min=0.03, max=1.0))
        prior_log = prior_log - torch.mean(prior_log, dim=-1, keepdim=True)
        gate_w = torch.clamp(self.knn_prior_weight, min=0.2, max=2.6)
        emissions_l2 = emissions_l2 + alpha_near * gate_w * prior_log

        # 若 L1 類別數 >= 6 且對應沉積粗分類 (2:細粒泥質, 3:砂質, 4:礫石, 5:互層)，同步注入近場粗分類先驗
        if self.n_classes_l1 >= 6 and self.n_classes_l2 == 6:
            p_l1_mud = p_knn[..., 0] + p_knn[..., 1]
            p_l1_sand = p_knn[..., 2] + p_knn[..., 3]
            p_l1_mixed = p_knn[..., 4]
            p_l1_gravel = p_knn[..., 5]
            l1_boost = torch.zeros_like(logits_l1)
            l1_boost[..., 2] = torch.log(torch.clamp(p_l1_mud, min=0.05))
            l1_boost[..., 3] = torch.log(torch.clamp(p_l1_sand, min=0.05))
            l1_boost[..., 4] = torch.log(torch.clamp(p_l1_gravel, min=0.05))
            l1_boost[..., 5] = torch.log(torch.clamp(p_l1_mixed, min=0.05))
            logits_l1 = logits_l1 + alpha_near * (0.5 * gate_w) * l1_boost

        return emissions_l2, logits_l1

    def apply_logit_adjustment(self, emissions_l2: torch.Tensor) -> torch.Tensor:
        """在送入 CRF 解碼前加入類別先驗補償 (Logit Adjustment)，確保少數類粉土 (Silt) Recall > 0%。"""
        return emissions_l2 + self.logit_adjustment.view(1, 1, -1)

    def forward(self, x: torch.Tensor, knn_prior: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        emissions_l2, logits_l1 = self.extract_emissions(x, knn_prior=knn_prior)
        return {
            "emissions_l2": emissions_l2,
            "logits_l1": logits_l1
        }

    def decode_l2(
        self,
        emissions_l2: torch.Tensor,
        mask: torch.Tensor | None = None,
        transition_weight: float = 0.06
    ) -> List[List[int]]:
        """加入類別先驗補償後，使用 CRF 維特比演算法解碼最優路徑。"""
        adjusted = self.apply_logit_adjustment(emissions_l2)
        return self.crf.decode(adjusted, mask=mask, transition_weight=transition_weight)

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        transition_weight: float = 0.06,
        use_crf: bool = True,
        knn_prior: torch.Tensor | None = None
    ) -> Dict[str, Any]:
        """推論介面：回傳 L2 類別序列、L1 類別、L2 各層機率與信心度。"""
        self.eval()
        with torch.no_grad():
            emissions_l2, logits_l1 = self.extract_emissions(x, knn_prior=knn_prior)
            adjusted_l2 = self.apply_logit_adjustment(emissions_l2)

            if use_crf:
                best_paths_l2 = self.crf.decode(
                    adjusted_l2,
                    mask=mask,
                    transition_weight=transition_weight,
                    suppress_isolated_spikes=True
                )
            else:
                argmax_preds = torch.argmax(adjusted_l2, dim=-1).cpu().numpy()
                best_paths_l2 = []
                seq_lens = mask.long().sum(dim=1).tolist() if mask is not None else [adjusted_l2.shape[1]] * adjusted_l2.shape[0]
                for b in range(len(seq_lens)):
                    t_len = seq_lens[b]
                    best_paths_l2.append(argmax_preds[b, :t_len].tolist())

            preds_l1 = torch.argmax(logits_l1, dim=-1).cpu().numpy()
            probs_l2 = F.softmax(adjusted_l2, dim=-1).cpu().numpy()
            conf_l2 = np.max(probs_l2, axis=-1)

        return {
            "l2_paths": best_paths_l2,
            "l1_preds": preds_l1,
            "l2_probs": probs_l2,
            "l2_conf": conf_l2
        }
