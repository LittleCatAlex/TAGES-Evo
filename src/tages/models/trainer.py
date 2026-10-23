"""井身序列神經網路訓練與評估器 (WellboreSeqNet Trainer & Evaluator)。"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from tages.models.losses import HierarchicalSequenceLoss
from tages.models.well_seq_net import WellboreSeqNet
from tages.features.sequence_features import SequenceFeatureExtractor


class WellboreSeqTrainer:
    """井身序列模型端到端訓練器。"""

    def __init__(
        self,
        model: WellboreSeqNet,
        extractor: SequenceFeatureExtractor,
        l2_classes: list[int] | None = None,
        l1_classes: list[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
        loss_fn: HierarchicalSequenceLoss | None = None,
        scheduler: Any | None = None
    ):
        self.device = torch.device(device)
        self.extractor = extractor
        if hasattr(model, "bind_extractor"):
            model.bind_extractor(extractor)
        self.model = model.to(self.device)
        self.l2_classes = l2_classes or [2, 3, 4, 5, 6, 7]
        self.l1_classes = l1_classes or [1, 2, 3, 4, 5, 6, 7]

        self.loss_fn = loss_fn or HierarchicalSequenceLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = scheduler
        self.best_macro_f1 = 0.0

    def apply_sequence_augmentation(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """訓練階段之地質序列資料增強：施加輕微速度微擾 (Velocity Jitter)，保留真實空間座標供多尺度傅立葉編碼。"""
        if not self.model.training:
            return x

        x_aug = x.clone()
        noise = torch.randn_like(x_aug[:, :, 3:5]) * 0.02
        x_aug[:, :, 3:5] = x_aug[:, :, 3:5] + noise
        return x_aug

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_crf = 0.0
        total_focal = 0.0
        total_phy = 0.0
        n_batches = len(dataloader)

        for batch in dataloader:
            x = batch["x"].to(self.device)
            knn_prior = batch["knn_prior"].to(self.device) if "knn_prior" in batch else None
            y_l2 = batch["y_l2"].to(self.device)
            y_l1 = batch["y_l1"].to(self.device)
            mask = batch["mask"].to(self.device)

            x = self.apply_sequence_augmentation(x, mask)

            self.optimizer.zero_grad()
            emissions_l2, logits_l1 = self.model.extract_emissions(x, knn_prior=knn_prior)

            losses = self.loss_fn(
                model=self.model,
                emissions_l2=emissions_l2,
                logits_l1=logits_l1,
                y_l2=y_l2,
                y_l1=y_l1,
                mask=mask,
                x=x
            )

            loss = losses["total_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_crf += losses["loss_crf"].item()
            total_focal += losses["loss_focal_l2"].item()
            if "loss_phy" in losses:
                total_phy += losses["loss_phy"].item()

        if self.scheduler is not None:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "loss": total_loss / max(1, n_batches),
            "loss_crf": total_crf / max(1, n_batches),
            "loss_focal": total_focal / max(1, n_batches),
            "loss_phy": total_phy / max(1, n_batches),
            "lr": current_lr
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        transition_weight: float = 0.06,
        use_crf: bool = True
    ) -> Dict[str, Any]:
        self.model.eval()
        all_true_l2 = []
        all_pred_l2 = []
        all_true_l1 = []
        all_pred_l1 = []

        total_transitions = 0
        isolated_noise_count = 0
        total_depth_m = 0.0

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(self.device)
                knn_prior = batch["knn_prior"].to(self.device) if "knn_prior" in batch else None
                y_l2 = batch["y_l2"].cpu().numpy()
                y_l1 = batch["y_l1"].cpu().numpy()
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]
                depths_batch = batch.get("depths")

                res = self.model.predict(
                    x,
                    mask=mask,
                    transition_weight=transition_weight,
                    use_crf=use_crf,
                    knn_prior=knn_prior
                )
                pred_paths = res["l2_paths"]
                preds_l1 = res["l1_preds"]

                for b in range(len(lengths)):
                    t_len = lengths[b]
                    true_seq_l2 = y_l2[b, :t_len]
                    pred_seq_l2 = pred_paths[b]
                    true_seq_l1 = y_l1[b, :t_len]
                    pred_seq_l1 = preds_l1[b, :t_len]

                    all_true_l2.extend(true_seq_l2)
                    all_pred_l2.extend(pred_seq_l2)
                    all_true_l1.extend(true_seq_l1)
                    all_pred_l1.extend(pred_seq_l1)

                    p = np.array(pred_seq_l2)
                    if len(p) > 1:
                        transitions = np.sum(p[:-1] != p[1:])
                        total_transitions += int(transitions)

                    if depths_batch is not None:
                        d_seq = depths_batch[b, :t_len].cpu().numpy()
                        span = float(d_seq[-1] - d_seq[0]) if len(d_seq) > 1 else 1.0
                        total_depth_m += max(1.0, span)
                    else:
                        total_depth_m += len(p) * 2.0

                    for k in range(1, len(p) - 1):
                        if p[k - 1] == p[k + 1] and p[k] != p[k - 1]:
                            isolated_noise_count += 1

        all_true_l2 = np.array(all_true_l2)
        all_pred_l2 = np.array(all_pred_l2)
        all_true_l1 = np.array(all_true_l1)
        all_pred_l1 = np.array(all_pred_l1)

        acc_l2 = accuracy_score(all_true_l2, all_pred_l2)
        macro_f1_l2 = f1_score(all_true_l2, all_pred_l2, average="macro", zero_division=0)
        weighted_f1_l2 = f1_score(all_true_l2, all_pred_l2, average="weighted", zero_division=0)

        acc_l1 = accuracy_score(all_true_l1, all_pred_l1)
        macro_f1_l1 = f1_score(all_true_l1, all_pred_l1, average="macro", zero_division=0)

        trans_per_100m = (total_transitions / max(1.0, total_depth_m)) * 100.0

        return {
            "accuracy_l2": acc_l2,
            "macro_f1_l2": macro_f1_l2,
            "weighted_f1_l2": weighted_f1_l2,
            "accuracy_l1": acc_l1,
            "macro_f1_l1": macro_f1_l1,
            "trans_per_100m": trans_per_100m,
            "isolated_noise_count": isolated_noise_count,
            "y_true_l2": all_true_l2,
            "y_pred_l2": all_pred_l2
        }

    def save_checkpoint(self, path: str | Path, extra_meta: dict | None = None):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "scaler_state": self.extractor.get_state(),
            "l2_classes": self.l2_classes,
            "l1_classes": self.l1_classes,
            "n_classes_l2": self.model.n_classes_l2,
            "n_classes_l1": self.model.n_classes_l1,
            "best_macro_f1": self.best_macro_f1,
            "in_features": self.model.in_features,
            "cnn_channels": self.model.conv_in.out_channels,
            "lstm_hidden": self.model.bilstm.hidden_size,
            "lstm_layers": self.model.bilstm.num_layers,
            "meta": extra_meta or {}
        }
        torch.save(ckpt, out_path)
        print(f"WellboreSeqNet 檢查點已儲存至: {out_path}")

    @staticmethod
    def load_checkpoint(path: str | Path, device: str = "cpu") -> Tuple[WellboreSeqNet, SequenceFeatureExtractor, dict]:
        in_path = Path(path)
        ckpt = torch.load(in_path, map_location=device, weights_only=False)

        extractor = SequenceFeatureExtractor()
        extractor.load_state(ckpt["scaler_state"])

        model = WellboreSeqNet(
            in_features=ckpt.get("in_features", 14),
            cnn_channels=ckpt.get("cnn_channels", 96),
            lstm_hidden=ckpt.get("lstm_hidden", 128),
            lstm_layers=ckpt.get("lstm_layers", 2),
            n_classes_l2=ckpt.get("n_classes_l2", len(ckpt["l2_classes"])),
            n_classes_l1=ckpt.get("n_classes_l1", len(ckpt["l1_classes"]))
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.bind_extractor(extractor)
        model.to(device)
        model.eval()
        return model, extractor, ckpt
