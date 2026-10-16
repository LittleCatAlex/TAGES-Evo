"""井身序列 PyTorch 資料集與批次動態對齊遮罩模組 (Wellbore Sequence PyTorch Dataset & Dynamic Padding Collation)。"""
from __future__ import annotations
from typing import Sequence, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tages.features.sequence_features import SequenceFeatureExtractor, FEATURE_NAMES

class WellboreSequenceDataset(Dataset):
    """將表格型地層數據依井位聚合為一維垂直時序樣本之 PyTorch Dataset。"""

    def __init__(
        self,
        df: pd.DataFrame,
        extractor: SequenceFeatureExtractor,
        l2_classes: list[int] | None = None,
        l1_classes: list[int] | None = None,
        station_col: str = "station_id",
        stride: int = 1
    ):
        self.extractor = extractor
        self.stride = stride
        self.l2_classes = l2_classes or [2, 3, 4, 5, 6, 7]
        self.l2_to_idx = {c: i for i, c in enumerate(self.l2_classes)}
        
        self.l1_classes = l1_classes or [0, 1, 2, 3, 4, 5, 6, 7]
        self.l1_to_idx = {c: i for i, c in enumerate(self.l1_classes)}

        # 過濾僅保留有效類別的紀錄
        df_valid = df[df["lithology_id"].isin(self.l2_to_idx.keys())].copy()
        
        # 依井位分組
        self.wells_data = []
        grouped = df_valid.groupby(station_col, sort=False)
        for st_id, grp in grouped:
            grp_sorted = grp.sort_values(by="depth_m").drop_duplicates(subset=["depth_m"]).reset_index(drop=True)
            if self.stride > 1:
                grp_sorted = grp_sorted.iloc[::self.stride].reset_index(drop=True)
            if len(grp_sorted) < 2:
                continue

            feats_norm = self.extractor.transform(grp_sorted) # (T, 14)
            y_l2 = np.array([self.l2_to_idx[l] for l in grp_sorted["lithology_id"].values], dtype=np.int64)
            
            l1_col = "lithology_l1_id" if "lithology_l1_id" in grp_sorted.columns else "lithology_id"
            y_l1 = np.array([self.l1_to_idx.get(l, 0) for l in grp_sorted[l1_col].values], dtype=np.int64)
            
            depths = grp_sorted["depth_m"].values.astype(np.float32)

            self.wells_data.append({
                "station_id": str(st_id),
                "x": feats_norm.astype(np.float32),
                "y_l2": y_l2,
                "y_l1": y_l1,
                "depths": depths,
                "length": len(depths)
            })

    def __len__(self) -> int:
        return len(self.wells_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.wells_data[idx]


def collate_well_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """將不同深度的井身序列動態 Padding 對齊為標準 3D Batch 張量。
    
    回傳：
        x: (Batch, Max_T, 14) FloatTensor
        y_l2: (Batch, Max_T) LongTensor
        y_l1: (Batch, Max_T) LongTensor
        mask: (Batch, Max_T) BoolTensor (True: 有效深度, False: Padding)
        depths: (Batch, Max_T) FloatTensor
        station_ids: list[str]
    """
    batch_size = len(batch)
    lengths = [b["length"] for b in batch]
    max_t = max(lengths)
    n_feats = batch[0]["x"].shape[1]

    x_padded = np.zeros((batch_size, max_t, n_feats), dtype=np.float32)
    y_l2_padded = np.zeros((batch_size, max_t), dtype=np.int64)
    y_l1_padded = np.zeros((batch_size, max_t), dtype=np.int64)
    depths_padded = np.zeros((batch_size, max_t), dtype=np.float32)
    mask = np.zeros((batch_size, max_t), dtype=bool)
    station_ids = []

    for i, item in enumerate(batch):
        t = item["length"]
        x_padded[i, :t, :] = item["x"]
        y_l2_padded[i, :t] = item["y_l2"]
        y_l1_padded[i, :t] = item["y_l1"]
        depths_padded[i, :t] = item["depths"]
        mask[i, :t] = True
        station_ids.append(item["station_id"])

    return {
        "x": torch.from_numpy(x_padded),
        "y_l2": torch.from_numpy(y_l2_padded),
        "y_l1": torch.from_numpy(y_l1_padded),
        "mask": torch.from_numpy(mask),
        "depths": torch.from_numpy(depths_padded),
        "station_ids": station_ids,
        "lengths": lengths
    }
