"""地球物理電測監督預訓練與知識特徵蒸餾模組 (Wireline Supervised Pretraining & Knowledge Distillation)。

核心架構：
1. 教師網路 (PetrophysicsTeacher): 接收 17 維特徵 (14 維時序物理 + 3 維實測電測)，結合近場空間錨定先驗輸出高純度岩性機率分佈。
2. 學生網路 (DistilledStudentNet): 僅需 14 維空間與震波特徵，內建固定基準深度 (z_norm, z_log)、多尺度空間傅立葉位置編碼、
   近場 K-NN / IDW 相律先驗門控、少數類 Logit Adjustment 以及虛擬電測輔助預測頭 (Pseudo I_GR & Pseudo log10(N64))。
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tages.models.crf import LinearChainCRF
from tages.models.well_seq_net import MultiScaleConv1DBlock


class PetrophysicsTeacher(nn.Module):
    """電測監督教師網路：以 17 維完整岩石物理特徵與近場空間錨定為輸入進行高精準岩相分類。"""

    def __init__(
        self,
        in_features: int = 17,
        cnn_channels: int = 64,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        n_classes_l2: int = 6,
        n_classes_l1: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes_l2 = n_classes_l2
        self.n_classes_l1 = n_classes_l1

        self.conv_in = nn.Conv1d(in_features, cnn_channels, kernel_size=1)
        self.aux_conv_in = nn.Conv1d(n_classes_l2 + 1, cnn_channels, kernel_size=1)
        nn.init.normal_(self.aux_conv_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.aux_conv_in.bias)

        self.block1 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)
        self.block2 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)

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

        self.head_l1 = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes_l1)
        )
        self.head_l2 = nn.Sequential(
            nn.Linear(bilstm_out_dim + n_classes_l1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes_l2)
        )
        self.knn_prior_weight = nn.Parameter(torch.tensor(1.25, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        knn_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        if knn_prior is None:
            knn_prior_t = torch.zeros((B, T, self.n_classes_l2 + 1), dtype=x.dtype, device=device)
            knn_prior_t[..., :self.n_classes_l2] = 1.0 / float(self.n_classes_l2)
        else:
            knn_prior_t = knn_prior.to(device)

        h = self.conv_in(x.transpose(1, 2)) + self.aux_conv_in(knn_prior_t.transpose(1, 2))
        h = self.block1(h)
        h = self.block2(h)

        h_seq = self.dropout(h.transpose(1, 2))
        lstm_out, _ = self.bilstm(h_seq)

        logits_l1 = self.head_l1(lstm_out)
        l2_in = torch.cat([lstm_out, logits_l1], dim=-1)
        logits_l2 = self.head_l2(l2_in)

        p_knn = knn_prior_t[..., :self.n_classes_l2]
        alpha_near = knn_prior_t[..., self.n_classes_l2:self.n_classes_l2 + 1]
        prior_log = torch.log(torch.clamp(p_knn, min=0.05, max=1.0))
        prior_log = prior_log - torch.mean(prior_log, dim=-1, keepdim=True)
        logits_l2 = logits_l2 + alpha_near * torch.clamp(self.knn_prior_weight, 0.2, 1.8) * prior_log

        return logits_l2, logits_l1, lstm_out


class DistilledStudentNet(nn.Module):
    """特徵蒸餾學生網路：僅需 14 維空間與震波特徵，具備虛擬電測輔助預測頭、近場空間先驗門控與瓦爾特相律 CRF。"""

    def __init__(
        self,
        in_features: int = 14,
        cnn_channels: int = 64,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        n_classes_l2: int = 6,
        n_classes_l1: int = 8,
        n_petro_targets: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes_l2 = n_classes_l2
        self.n_classes_l1 = n_classes_l1
        self.n_petro_targets = n_petro_targets
        self._bound_extractor = None

        default_mean = np.zeros(max(14, in_features), dtype=np.float32)
        default_mean[0] = 45.0
        default_scale = np.ones(max(14, in_features), dtype=np.float32)
        default_scale[0] = 35.0
        self.register_buffer("scaler_mean", torch.from_numpy(default_mean))
        self.register_buffer("scaler_scale", torch.from_numpy(default_scale))

        default_adj = np.zeros(n_classes_l2, dtype=np.float32)
        if n_classes_l2 == 6:
            default_adj = np.array([0.0, 0.22, 0.0, 0.05, 0.05, 0.0], dtype=np.float32)
        self.register_buffer("logit_adjustment", torch.from_numpy(default_adj))

        self.aux_dim = 14 + n_classes_l2 + 1
        self.conv_in = nn.Conv1d(in_features, cnn_channels, kernel_size=1)
        self.aux_conv_in = nn.Conv1d(self.aux_dim, cnn_channels, kernel_size=1)
        nn.init.normal_(self.aux_conv_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.aux_conv_in.bias)

        self.block1 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)
        self.block2 = MultiScaleConv1DBlock(cnn_channels, cnn_channels)

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

        self.head_l1 = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes_l1)
        )
        self.head_l2 = nn.Sequential(
            nn.Linear(bilstm_out_dim + n_classes_l1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes_l2)
        )

        # 輔助虛擬電測回歸頭 (結合雙向時序特徵與跨尺度物理特徵預測 I_GR 與 log10(N64))
        self.head_petro = nn.Sequential(
            nn.Linear(bilstm_out_dim + in_features + self.aux_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, n_petro_targets)
        )

        self.knn_prior_weight = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.crf = LinearChainCRF(num_tags=n_classes_l2)

    def bind_extractor(self, extractor: Any):
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

    def set_class_priors(self, priors: np.ndarray | Sequence[float], tau: float = 0.18):
        p = np.asarray(priors, dtype=np.float32)
        if len(p) != self.n_classes_l2:
            return
        p = np.clip(p / np.maximum(np.sum(p), 1e-6), 1e-4, 1.0)
        mean_p = float(np.mean(p))
        adj = np.clip(np.maximum(0.0, float(tau) * np.log(mean_p / p)), 0.0, 0.32).astype(np.float32)
        with torch.no_grad():
            self.logit_adjustment.copy_(torch.from_numpy(adj).to(self.logit_adjustment.device))

    def _compute_aux_features(
        self,
        x: torch.Tensor,
        knn_prior: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device = x.device
        depth_m = torch.clamp(x[..., 0] * self.scaler_scale[0] + self.scaler_mean[0], min=0.0)
        z_norm = torch.clamp(depth_m / 350.0, 0.0, 1.0).unsqueeze(-1)
        z_log = (torch.log1p(depth_m) / math.log1p(350.0)).unsqueeze(-1)

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
        spatial_enc = torch.cat(sin_cos_list, dim=-1)

        if knn_prior is not None:
            knn_prior_t = knn_prior.to(device)
        elif self._bound_extractor is not None and getattr(self._bound_extractor, "has_spatial_prior", False):
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

    def forward(
        self,
        x: torch.Tensor,
        knn_prior: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        aux_feats, p_knn, alpha_near = self._compute_aux_features(x, knn_prior=knn_prior)

        h = F.relu(self.conv_in(x.transpose(1, 2)) + self.aux_conv_in(aux_feats.transpose(1, 2)))
        h = self.block1(h)
        h = self.block2(h)

        h_seq = self.dropout(h.transpose(1, 2))
        lstm_out, _ = self.bilstm(h_seq)

        logits_l1 = self.head_l1(lstm_out)
        l2_in = torch.cat([lstm_out, logits_l1], dim=-1)
        logits_l2 = self.head_l2(l2_in)

        prior_log = torch.log(torch.clamp(p_knn, min=0.05, max=1.0))
        prior_log = prior_log - torch.mean(prior_log, dim=-1, keepdim=True)
        gate_w = torch.clamp(self.knn_prior_weight, min=0.2, max=1.8)
        logits_l2 = logits_l2 + alpha_near * gate_w * prior_log

        petro_in = torch.cat([lstm_out, x, aux_feats], dim=-1)
        pred_petro = self.head_petro(petro_in)

        return {
            "logits_l2": logits_l2,
            "logits_l1": logits_l1,
            "pred_petro": pred_petro,
            "lstm_out": lstm_out
        }

    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transition_weight: float = 0.06,
        use_crf: bool = True,
        knn_prior: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """相容 WellboreSeqNet 推論介面"""
        self.eval()
        with torch.no_grad():
            res = self.forward(x, knn_prior=knn_prior)
            emissions = res["logits_l2"] + self.logit_adjustment.view(1, 1, -1)
            logits_l1 = res["logits_l1"]
            pred_petro = res["pred_petro"]

            probs_l2 = F.softmax(emissions, dim=-1).cpu().numpy()
            preds_l1 = torch.argmax(logits_l1, dim=-1).cpu().numpy()

            if use_crf:
                paths = self.crf.decode(
                    emissions,
                    mask=mask,
                    transition_weight=transition_weight,
                    suppress_isolated_spikes=True
                )
                scores = [0.0] * x.shape[0]
            else:
                best_idx = torch.argmax(emissions, dim=-1)
                paths = [best_idx[b].cpu().numpy().tolist() for b in range(x.shape[0])]
                scores = [0.0] * x.shape[0]

            conf_list = []
            for b in range(len(paths)):
                p_seq = paths[b]
                b_confs = [float(probs_l2[b, t, p_seq[t]]) for t in range(len(p_seq))]
                conf_list.append(b_confs)

            return {
                "l2_paths": paths,
                "l2_scores": scores,
                "l2_conf": conf_list,
                "l2_probs": probs_l2,
                "l1_preds": preds_l1,
                "pred_petro": pred_petro.cpu().numpy()
            }


class DistillationLoss(nn.Module):
    """特徵與知識蒸餾多任務損失函數。"""

    def __init__(
        self,
        temperature: float = 3.0,
        weight_ce: float = 1.0,
        weight_kd: float = 0.8,
        weight_aux_petro: float = 0.5,
        weight_crf: float = 0.15
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_ce = weight_ce
        self.weight_kd = weight_kd
        self.weight_aux_petro = weight_aux_petro
        self.weight_crf = weight_crf
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        student_logits_l2: torch.Tensor,
        teacher_logits_l2: Optional[torch.Tensor],
        student_pred_petro: torch.Tensor,
        true_petro: Optional[torch.Tensor],
        targets_l2: torch.Tensor,
        mask: torch.Tensor,
        petro_mask: Optional[torch.Tensor] = None,
        crf_module: Optional[LinearChainCRF] = None
    ) -> Dict[str, torch.Tensor]:
        B, T, C = student_logits_l2.shape
        flat_mask = mask.view(-1)
        total_valid = flat_mask.sum().clamp(min=1.0)

        ce_per_token = self.ce_loss(
            student_logits_l2.view(-1, C),
            targets_l2.view(-1)
        )
        loss_ce = (ce_per_token * flat_mask).sum() / total_valid

        if teacher_logits_l2 is not None:
            t = self.temperature
            p_teacher = F.softmax(teacher_logits_l2 / t, dim=-1)
            log_p_student = F.log_softmax(student_logits_l2 / t, dim=-1)
            kl_div = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)

            if petro_mask is not None:
                kd_mask = mask & petro_mask
            else:
                kd_mask = mask
            loss_kd = (kl_div * kd_mask.float()).sum() / kd_mask.sum().clamp(min=1.0) * (t ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits_l2.device)

        if true_petro is not None and petro_mask is not None and petro_mask.any():
            mse_per_token = self.mse_loss(student_pred_petro, true_petro).mean(dim=-1)
            valid_p = (mask & petro_mask).float()
            loss_petro = (mse_per_token * valid_p).sum() / valid_p.sum().clamp(min=1.0)
        else:
            loss_petro = torch.tensor(0.0, device=student_logits_l2.device)

        if crf_module is not None:
            if hasattr(crf_module, "forward_loss"):
                crf_nll = crf_module.forward_loss(student_logits_l2, targets_l2, mask=mask)
            else:
                crf_nll = crf_module(student_logits_l2, targets_l2, mask=mask)
            loss_crf = crf_nll
        else:
            loss_crf = torch.tensor(0.0, device=student_logits_l2.device)

        total_loss = (
            self.weight_ce * loss_ce
            + self.weight_kd * loss_kd
            + self.weight_aux_petro * loss_petro
            + self.weight_crf * loss_crf
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_petro": loss_petro,
            "loss_crf": loss_crf
        }
