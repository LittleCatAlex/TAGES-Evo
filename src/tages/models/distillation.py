"""地球物理電測監督預訓練與知識特徵蒸餾模組 (Wireline Supervised Pretraining & Knowledge Distillation)。

核心動機：
1. 384 口深水觀測井擁有 130 萬筆連續地球物理測井 (自然伽瑪 I_GR、視電阻率 N64、泥質含量 V_shale)，
   具備直接區分泥/砂/礫石的高解析物理先驗。
2. 盆地三維空間網格 (3D Voxels) 僅有震波層析波速 (Vp, Vs) 與空間坐標，缺乏井下電測儀器數據。
3. 透過教師-學生特徵蒸餾架構：
   - 教師網路 (Teacher): 接收 17 維特徵 (14 維時序物理 + 3 維實測電測)，輸出高純度岩性機率分佈。
   - 學生網路 (Student): 僅接收 14 維時序物理特徵，透過 KL 散度蒸餾教師機率分佈，
     並利用多任務輔助頭預測「虛擬電測」(Pseudo I_GR & Pseudo log(N64))。
   - 使得學生網路在全臺任意 3D 體素節點推論時，皆能調用蒸餾自百萬筆電測資料庫的岩石物理知識！
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tages.models.crf import LinearChainCRF
from tages.models.well_seq_net import MultiScaleConv1DBlock


class PetrophysicsTeacher(nn.Module):
    """電測監督教師網路：以 17 維完整岩石物理特徵為輸入進行高精準岩相分類。"""

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
        bilstm_out_dim = lstm_hidden * 2  # 192

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (Batch, Seq_T, 17)
        回傳:
          logits_l2: (Batch, Seq_T, n_classes_l2)
          logits_l1: (Batch, Seq_T, n_classes_l1)
          h_feat: (Batch, Seq_T, bilstm_out_dim)
        """
        # (B, T, C) -> (B, C, T)
        h = x.transpose(1, 2)
        h = self.conv_in(h)
        h = self.block1(h)
        h = self.block2(h)

        # (B, C, T) -> (B, T, C)
        h_seq = h.transpose(1, 2)
        h_seq = self.dropout(h_seq)
        lstm_out, _ = self.bilstm(h_seq)

        logits_l1 = self.head_l1(lstm_out)
        l2_in = torch.cat([lstm_out, logits_l1], dim=-1)
        logits_l2 = self.head_l2(l2_in)

        return logits_l2, logits_l1, lstm_out


class DistilledStudentNet(nn.Module):
    """特徵蒸餾學生網路：僅需 14 維空間與震波特徵，具備虛擬電測輔助預測頭與瓦爾特相律 CRF。"""

    def __init__(
        self,
        in_features: int = 14,
        cnn_channels: int = 64,
        lstm_hidden: int = 96,
        lstm_layers: int = 2,
        n_classes_l2: int = 6,
        n_classes_l1: int = 8,
        n_petro_targets: int = 2,  # 預測 I_GR 與 log10(N64)
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.n_classes_l2 = n_classes_l2
        self.n_classes_l1 = n_classes_l1
        self.n_petro_targets = n_petro_targets

        self.conv_in = nn.Conv1d(in_features, cnn_channels, kernel_size=1)
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

        # 輔助虛擬電測回歸頭 (Reconstructs pseudo-Gamma & pseudo-Resistivity)
        self.head_petro = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_petro_targets)
        )

        # 瓦爾特相律條件隨機場
        self.crf = LinearChainCRF(num_tags=n_classes_l2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (Batch, Seq_T, 14)
        回傳:
          logits_l2: (Batch, Seq_T, n_classes_l2)
          logits_l1: (Batch, Seq_T, n_classes_l1)
          pred_petro: (Batch, Seq_T, 2)
          lstm_out: (Batch, Seq_T, dim)
        """
        h = x.transpose(1, 2)
        h = self.conv_in(h)
        h = self.block1(h)
        h = self.block2(h)

        h_seq = h.transpose(1, 2)
        h_seq = self.dropout(h_seq)
        lstm_out, _ = self.bilstm(h_seq)

        logits_l1 = self.head_l1(lstm_out)
        l2_in = torch.cat([lstm_out, logits_l1], dim=-1)
        logits_l2 = self.head_l2(l2_in)

        pred_petro = self.head_petro(lstm_out)

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
        transition_weight: float = 0.20,
        use_crf: bool = True
    ) -> Dict[str, Any]:
        """相容 WellboreSeqNet 推論介面"""
        self.eval()
        with torch.no_grad():
            res = self.forward(x)
            emissions = res["logits_l2"]
            logits_l1 = res["logits_l1"]
            pred_petro = res["pred_petro"]

            probs_l2 = F.softmax(emissions, dim=-1).cpu().numpy()
            preds_l1 = torch.argmax(logits_l1, dim=-1).cpu().numpy()

            if use_crf:
                paths = self.crf.decode(emissions, mask=mask, transition_weight=transition_weight)
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
        weight_crf: float = 0.08
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
        """
        student_logits_l2: (B, T, C)
        teacher_logits_l2: (B, T, C) or None
        student_pred_petro: (B, T, 2)
        true_petro: (B, T, 2) or None
        targets_l2: (B, T)
        mask: (B, T) bool
        petro_mask: (B, T) bool (樣本是否具備實測電測值)
        """
        B, T, C = student_logits_l2.shape
        flat_mask = mask.view(-1)
        total_valid = flat_mask.sum().clamp(min=1.0)

        # 1. 地質標籤交叉熵硬損失 (Hard Cross-Entropy Loss)
        ce_per_token = self.ce_loss(
            student_logits_l2.view(-1, C),
            targets_l2.view(-1)
        )
        loss_ce = (ce_per_token * flat_mask).sum() / total_valid

        # 2. 教師網路機率分佈軟蒸餾 (KL Divergence Soft Loss)
        if teacher_logits_l2 is not None:
            t = self.temperature
            p_teacher = F.softmax(teacher_logits_l2 / t, dim=-1)
            log_p_student = F.log_softmax(student_logits_l2 / t, dim=-1)
            kl_div = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)  # (B, T)

            if petro_mask is not None:
                # 優先在具有電測監督之井位施加強蒸餾
                kd_mask = mask & petro_mask
            else:
                kd_mask = mask
            loss_kd = (kl_div * kd_mask.float()).sum() / kd_mask.sum().clamp(min=1.0) * (t ** 2)
        else:
            loss_kd = torch.tensor(0.0, device=student_logits_l2.device)

        # 3. 虛擬電測輔助預測損失 (Auxiliary Pseudo-Petrophysics MSE Loss)
        if true_petro is not None and petro_mask is not None and petro_mask.any():
            mse_per_token = self.mse_loss(student_pred_petro, true_petro).mean(dim=-1)  # (B, T)
            valid_p = (mask & petro_mask).float()
            loss_petro = (mse_per_token * valid_p).sum() / valid_p.sum().clamp(min=1.0)
        else:
            loss_petro = torch.tensor(0.0, device=student_logits_l2.device)

        # 4. CRF 序列層序損失
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
