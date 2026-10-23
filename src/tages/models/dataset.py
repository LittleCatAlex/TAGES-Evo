"""井身序列 PyTorch 資料集與批次動態對齊遮罩模組 (Wellbore Sequence PyTorch Dataset & Dynamic Padding Collation)。"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tages.features.sequence_features import (
    SequenceFeatureExtractor,
    FEATURE_NAMES,
    SEDIMENTARY_FEATURE_NAMES,
    compute_column_sequence_features
)


class WellboreSequenceDataset(Dataset):
    """將表格型地層數據依井位聚合為一維垂直時序樣本之 PyTorch Dataset。
    內建嚴格防洩漏 (Zero-Leakage, 排除自身 station_id) 之近場 K-NN / IDW 岩相先驗計算。
    """

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

        df_valid = df[df["lithology_id"].isin(self.l2_to_idx.keys())].copy()
        if len(df_valid) == 0:
            self.wells_data = []
            return

        # 一次性向量化排序與去重，避免在數千口井迴圈內重複呼叫 pandas DataFrame 操作
        df_valid = df_valid.sort_values(by=[station_col, "depth_m"], kind="mergesort").drop_duplicates(subset=[station_col, "depth_m"]).reset_index(drop=True)
        include_sed = any(c in self.extractor.feature_names for c in SEDIMENTARY_FEATURE_NAMES)

        has_pre_enriched = "_seq_enriched_v2" in df_valid.columns and all(c in df_valid.columns for c in self.extractor.feature_names)
        if has_pre_enriched:
            all_feats_norm = self.extractor.scaler.transform(df_valid[self.extractor.feature_names].values).astype(np.float32)
        else:
            all_feats_norm = None

        all_depths = df_valid["depth_m"].values.astype(np.float32)
        all_x97 = df_valid["x_twd97"].values.astype(np.float32) if "x_twd97" in df_valid.columns else np.full(len(df_valid), 200000.0, dtype=np.float32)
        all_y97 = df_valid["y_twd97"].values.astype(np.float32) if "y_twd97" in df_valid.columns else np.full(len(df_valid), 2600000.0, dtype=np.float32)
        all_l2 = np.array([self.l2_to_idx[l] for l in df_valid["lithology_id"].values], dtype=np.int64)
        l1_col = "lithology_l1_id" if "lithology_l1_id" in df_valid.columns else "lithology_id"
        all_l1 = np.array([self.l1_to_idx.get(l, 0) for l in df_valid[l1_col].values], dtype=np.int64)

        self.wells_data = []
        grouped_indices = df_valid.groupby(station_col, sort=False).indices
        for st_id, idx_arr in grouped_indices.items():
            if self.stride > 1:
                idx_arr = idx_arr[::self.stride]
            if len(idx_arr) < 2:
                continue
            if len(idx_arr) > 250:
                idx_arr = idx_arr[:250]

            depths = all_depths[idx_arr]
            x97 = float(all_x97[idx_arr[0]])
            y97 = float(all_y97[idx_arr[0]])

            if all_feats_norm is not None:
                feats_norm = all_feats_norm[idx_arr]
            else:
                grp_sorted = df_valid.iloc[idx_arr]
                feats_df = compute_column_sequence_features(
                    depths=depths,
                    z_msl=grp_sorted["z_msl"].values,
                    vp=grp_sorted["vp_ms"].values,
                    vs=grp_sorted["vs_ms"].values,
                    vp_vs=grp_sorted["vp_vs_ratio"].values,
                    poisson=grp_sorted["poisson_ratio"].values,
                    x_twd97=x97,
                    y_twd97=y97,
                    dist_to_fault_km=float(grp_sorted["dist_to_fault_km"].iloc[0]) if "dist_to_fault_km" in grp_sorted.columns else None,
                    dist_to_coast_km=float(grp_sorted["dist_to_coast_km"].iloc[0]) if "dist_to_coast_km" in grp_sorted.columns else None,
                    include_sedimentary=include_sed
                )
                feats_norm = self.extractor.scaler.transform(feats_df[self.extractor.feature_names].values).astype(np.float32)

            # 嚴格排除自身 station_id 之 Out-of-Fold 近場岩相先驗
            knn_prior = self.extractor.compute_knn_prior_for_column(
                depths=depths,
                x_twd97=x97,
                y_twd97=y97,
                exclude_station_id=str(st_id)
            )

            self.wells_data.append({
                "station_id": str(st_id),
                "x": feats_norm,
                "knn_prior": knn_prior.astype(np.float32),
                "y_l2": all_l2[idx_arr],
                "y_l1": all_l1[idx_arr],
                "depths": depths,
                "length": len(depths)
            })

        # 預設按序列長度分桶排序，避免淺孔 (T~10) 與深井 (T~150) 混批產生 85% Zero-Padding 污染 BatchNorm1d
        self.wells_data.sort(key=lambda w: w["length"])

    def shuffle_buckets(self, bucket_size: int = 64, seed: int = 42):
        """先按序列長度分桶形成同質長度批次 (消除 Padding 污染)，再將各批次順序隨機打亂，避免淺孔與深井梯度偏置。"""
        rng = np.random.default_rng(seed)
        self.wells_data.sort(key=lambda w: (w["length"] // 4, float(rng.random())))
        n = len(self.wells_data)
        chunks = [self.wells_data[i:min(i + bucket_size, n)] for i in range(0, n, bucket_size)]
        perm = rng.permutation(len(chunks))
        reordered = []
        for idx in perm:
            reordered.extend(chunks[idx])
        self.wells_data = reordered

    def __len__(self) -> int:
        return len(self.wells_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.wells_data[idx]


def collate_well_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """將不同深度的井身序列動態 Padding 對齊為標準 3D Batch 張量。"""
    batch_size = len(batch)
    lengths = [b["length"] for b in batch]
    max_t = max(lengths)
    n_feats = batch[0]["x"].shape[1]
    n_prior = batch[0]["knn_prior"].shape[1] if "knn_prior" in batch[0] else 7

    x_padded = np.zeros((batch_size, max_t, n_feats), dtype=np.float32)
    knn_prior_padded = np.zeros((batch_size, max_t, n_prior), dtype=np.float32)
    y_l2_padded = np.zeros((batch_size, max_t), dtype=np.int64)
    y_l1_padded = np.zeros((batch_size, max_t), dtype=np.int64)
    depths_padded = np.zeros((batch_size, max_t), dtype=np.float32)
    mask = np.zeros((batch_size, max_t), dtype=bool)
    station_ids = []

    for i, item in enumerate(batch):
        t = item["length"]
        x_padded[i, :t, :] = item["x"]
        if "knn_prior" in item:
            knn_prior_padded[i, :t, :] = item["knn_prior"]
        y_l2_padded[i, :t] = item["y_l2"]
        y_l1_padded[i, :t] = item["y_l1"]
        depths_padded[i, :t] = item["depths"]
        mask[i, :t] = True
        station_ids.append(item["station_id"])

    return {
        "x": torch.from_numpy(x_padded),
        "knn_prior": torch.from_numpy(knn_prior_padded),
        "y_l2": torch.from_numpy(y_l2_padded),
        "y_l1": torch.from_numpy(y_l1_padded),
        "mask": torch.from_numpy(mask),
        "depths": torch.from_numpy(depths_padded),
        "station_ids": station_ids,
        "lengths": lengths
    }
