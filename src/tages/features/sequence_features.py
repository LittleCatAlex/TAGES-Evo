"""井身深度序列特徵計算與空間正規化模組 (Wellbore Sequence Feature Engineering)。

計算每口井沿垂直深度軸演進的 14 維物理特徵：
1. depth_m: 深度 (m)
2. z_msl: 海拔高程 (m)
3. rel_depth_ratio: 跨尺度對數深度歸一化特徵 z_log = log(1 + z) / log(1 + 350.0)
4. vp_ms: 壓縮波速 (m/s)
5. vs_ms: 剪力波速 (m/s)
6. vp_vs_ratio: 縱橫波速比
7. poisson_ratio: 泊松比
8. dvs_dz_1m: 1m 局部剪力波速梯度
9. dvs_dz_5m: 5m 視窗波速梯度 (宏觀構造轉折)
10. d2vs_dz2: 剪力波速二階導數 (地層界面敏感指標)
11. vs_local_mean_9m: 上下 9m 局部波速滑動均值
12. vs_local_std_9m: 上下 9m 局部波速滑動標準差 (辨識均質厚層 vs. 砂泥互層)
13. x_twd97_norm: 正規化 TWD97 X 座標
14. y_twd97_norm: 正規化 TWD97 Y 座標

並提供：
- 固定物理基準深度歸一化 (z_norm = min(z / 350.0, 1.0)) 與對數深度特徵 (z_log = log1p(z) / log1p(350.0))
- 多尺度空間傅立葉正弦位置編碼 (Spatial Sinusoidal Encoding, 波長 0.5km, 2km, 10km)
- 嚴格防洩漏 (Zero-Leakage, 排除自身 station_id 與測試折) 之近場 K-NN / IDW 水平相律先驗
"""
from __future__ import annotations
from typing import Sequence
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "depth_m",
    "z_msl",
    "rel_depth_ratio",
    "vp_ms",
    "vs_ms",
    "vp_vs_ratio",
    "poisson_ratio",
    "dvs_dz_1m",
    "dvs_dz_5m",
    "d2vs_dz2",
    "vs_local_mean_9m",
    "vs_local_std_9m",
    "x_twd97_norm",
    "y_twd97_norm"
]

SEDIMENTARY_FEATURE_NAMES = [
    "dist_to_fault_km",
    "dist_to_coast_km",
    "vs_residual_ms"
]

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + SEDIMENTARY_FEATURE_NAMES

REF_MAX_DEPTH_M = 350.0
REF_DEPTH_STEP_M = 2.0
REF_N_DEPTH_BINS = int(REF_MAX_DEPTH_M // REF_DEPTH_STEP_M) + 1  # 176 bins (0m..350m)


def compute_depth_scale_features(depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """計算固定物理基準深度歸一化 (z_norm) 與對數深度特徵 (z_log)。
    消除單井 z / z_max 在 200m 水文深井與 20m 工程淺孔間的尺度錯位，
    並放大淺表 0-40m 工程鑽孔的垂直解析度。
    """
    z_safe = np.maximum(0.0, np.asarray(depths, dtype=float))
    z_norm = np.minimum(z_safe / REF_MAX_DEPTH_M, 1.0)
    z_log = np.log1p(z_safe) / np.log1p(REF_MAX_DEPTH_M)
    return z_norm, z_log


def compute_spatial_sinusoidal_encoding(
    x_twd97: np.ndarray | float,
    y_twd97: np.ndarray | float,
    wavelengths_m: tuple[float, ...] = (500.0, 2000.0, 10000.0)
) -> np.ndarray:
    """計算多尺度空間傅立葉位置編碼 (Spatial Sinusoidal Encoding, 波長 0.5km, 2km, 10km)。
    克服神經網路對平滑正規化座標的頻譜偏置 (Spectral Bias)。
    回傳形狀: (..., 4 * len(wavelengths_m))。
    """
    dx = np.asarray(x_twd97, dtype=float) - 200000.0
    dy = np.asarray(y_twd97, dtype=float) - 2600000.0
    enc_list = []
    for wl in wavelengths_m:
        omega = 2.0 * math.pi / float(wl)
        enc_list.append(np.sin(dx * omega))
        enc_list.append(np.cos(dx * omega))
        enc_list.append(np.sin(dy * omega))
        enc_list.append(np.cos(dy * omega))
    return np.stack(enc_list, axis=-1)


def compute_compaction_vs(
    depth_m: np.ndarray,
    vs0: float = 160.0,
    z0: float = 8.0,
    alpha: float = 0.32
) -> np.ndarray:
    """依據深度壓密冪次律計算正常壓實背景剪力波速 (m/s)。
    Vs_comp(z) = Vs0 * (1 + z / z0)^alpha
    """
    d_safe = np.maximum(0.0, np.asarray(depth_m, dtype=float))
    return np.round(vs0 * (1.0 + d_safe / z0) ** alpha, 1)


def compute_vs_residual(vs_ms: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """計算實測/層析剪力波速相對於正常壓實背景之殘差波速 (m/s)。"""
    vs_arr = np.asarray(vs_ms, dtype=float)
    vs_comp = compute_compaction_vs(depth_m)
    return np.round(vs_arr - vs_comp, 1)


def compute_column_sequence_features(
    depths: np.ndarray,
    z_msl: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    vp_vs: np.ndarray,
    poisson: np.ndarray,
    x_twd97: float,
    y_twd97: float,
    max_depth: float | None = None,
    dist_to_fault_km: float | None = None,
    dist_to_coast_km: float | None = None,
    include_sedimentary: bool = False
) -> pd.DataFrame:
    """針對單一垂直柱/單一井位計算沿深度軸演進的特徵 DataFrame。
    採用固定物理基準深度歸一化與對數深度特徵 (移除單井 z / z_max 錯位)。
    """
    n = len(depths)
    z_norm, z_log = compute_depth_scale_features(depths)
    # 以固定物理基準對數深度取代原本的單井相對深度 z / z_max
    rel_depth = z_log

    # 1. 計算梯度與二階導數 (確保深度單調嚴格遞增，防止重複深度導致除以零)
    depths_safe = depths.copy().astype(float)
    if len(depths_safe) > 1:
        for i in range(1, len(depths_safe)):
            if depths_safe[i] <= depths_safe[i - 1]:
                depths_safe[i] = depths_safe[i - 1] + 1e-3

    d_step = np.diff(depths_safe)
    mean_step = float(np.mean(d_step)) if len(d_step) > 0 else 1.0
    mean_step = max(0.2, mean_step)

    # 1m 局部梯度
    dvs_1m = np.gradient(vs, depths_safe) if n > 1 else np.zeros_like(vs)
    dvs_1m = np.nan_to_num(dvs_1m, nan=0.0, posinf=0.0, neginf=0.0)

    # 5m 視窗梯度 (平滑後梯度)
    k_5 = max(1, int(round(5.0 / mean_step)))
    vs_series = pd.Series(vs)
    vs_smooth_5 = vs_series.rolling(window=k_5, min_periods=1, center=True).mean().values
    dvs_5m = np.gradient(vs_smooth_5, depths_safe) if n > 1 else np.zeros_like(vs)
    dvs_5m = np.nan_to_num(dvs_5m, nan=0.0, posinf=0.0, neginf=0.0)

    # 二階導數
    d2vs = np.gradient(dvs_1m, depths_safe) if n > 1 else np.zeros_like(vs)
    d2vs = np.nan_to_num(d2vs, nan=0.0, posinf=0.0, neginf=0.0)

    # 上下 9m 滑動均值與標準差
    k_9 = max(1, int(round(9.0 / mean_step)))
    vs_mean_9m = vs_series.rolling(window=k_9, min_periods=1, center=True).mean().values
    vs_std_9m = vs_series.rolling(window=k_9, min_periods=1, center=True).std().fillna(0.0).values

    # 坐標正規化 (臺灣全島基準：X 基準 200,000m, Y 基準 2,600,000m, 尺度 100,000m)
    x_norm = np.full(n, (x_twd97 - 200000.0) / 100000.0, dtype=float)
    y_norm = np.full(n, (y_twd97 - 2600000.0) / 100000.0, dtype=float)

    data = {
        "depth_m": depths,
        "z_msl": z_msl,
        "rel_depth_ratio": rel_depth,
        "vp_ms": vp,
        "vs_ms": vs,
        "vp_vs_ratio": vp_vs,
        "poisson_ratio": poisson,
        "dvs_dz_1m": dvs_1m,
        "dvs_dz_5m": dvs_5m,
        "d2vs_dz2": d2vs,
        "vs_local_mean_9m": vs_mean_9m,
        "vs_local_std_9m": vs_std_9m,
        "x_twd97_norm": x_norm,
        "y_twd97_norm": y_norm
    }

    if include_sedimentary:
        d_fault = dist_to_fault_km if dist_to_fault_km is not None else 10.0
        d_coast = dist_to_coast_km if dist_to_coast_km is not None else 15.0
        data["dist_to_fault_km"] = np.full(n, float(d_fault), dtype=float)
        data["dist_to_coast_km"] = np.full(n, float(d_coast), dtype=float)
        data["vs_residual_ms"] = compute_vs_residual(vs, depths)
        cols = EXTENDED_FEATURE_NAMES
    else:
        cols = FEATURE_NAMES

    return pd.DataFrame(data)[cols]


class SequenceFeatureExtractor:
    """序列特徵標準化與近場空間錨定先驗提取器，支援 fit(train_df) 與 transform(df)。"""

    def __init__(self, feature_names: Sequence[str] | None = None, l2_classes: Sequence[int] | None = None):
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        self.l2_classes = list(l2_classes) if l2_classes is not None else [2, 3, 4, 5, 6, 7]
        self.l2_to_idx = {c: i for i, c in enumerate(self.l2_classes)}
        self.scaler = StandardScaler()
        self.fitted = False

        # 近場空間錨定先驗 (Out-of-Fold K-NN / IDW 水平相律先驗)
        self.has_spatial_prior = False
        self.ref_station_ids: np.ndarray | None = None
        self.ref_xy: np.ndarray | None = None
        self.ref_well_classes: np.ndarray | None = None  # (M_wells, REF_N_DEPTH_BINS) int8, -1 為無觀測
        self.global_depth_prior: np.ndarray | None = None  # (REF_N_DEPTH_BINS, n_classes) float32
        self.class_priors: np.ndarray | None = None  # (n_classes,) float32
        self._kd_tree: cKDTree | None = None

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """依井位 group 計算各井的時序衍生特徵並附加回 DataFrame。"""
        if "_seq_enriched_v2" in df.columns and all(c in df.columns for c in self.feature_names):
            return df

        station_col = "station_id" if "station_id" in df.columns else ("station" if "station" in df.columns else None)
        include_sed = any(c in self.feature_names for c in SEDIMENTARY_FEATURE_NAMES)

        if station_col and df[station_col].nunique() > 1:
            res_df = df.copy()
            grouped_indices = res_df.groupby(station_col, sort=False).indices

            all_d = res_df["depth_m"].values
            all_z = res_df["z_msl"].values
            all_vp = res_df["vp_ms"].values
            all_vs = res_df["vs_ms"].values
            all_vp_vs = res_df["vp_vs_ratio"].values
            all_pois = res_df["poisson_ratio"].values
            all_x = res_df["x_twd97"].values
            all_y = res_df["y_twd97"].values
            has_fault = "dist_to_fault_km" in res_df.columns
            has_coast = "dist_to_coast_km" in res_df.columns
            all_fault = res_df["dist_to_fault_km"].values if has_fault else None
            all_coast = res_df["dist_to_coast_km"].values if has_coast else None

            target_cols = EXTENDED_FEATURE_NAMES if include_sed else FEATURE_NAMES
            out_mat = np.zeros((len(res_df), len(target_cols)), dtype=np.float64)

            for _, idx_arr in grouped_indices.items():
                d_sub = all_d[idx_arr]
                if len(d_sub) > 1 and np.any(np.diff(d_sub) < 0):
                    order = np.argsort(d_sub, kind="mergesort")
                    idx_ord = idx_arr[order]
                else:
                    idx_ord = idx_arr
                i0 = idx_ord[0]
                feats = compute_column_sequence_features(
                    depths=all_d[idx_ord],
                    z_msl=all_z[idx_ord],
                    vp=all_vp[idx_ord],
                    vs=all_vs[idx_ord],
                    vp_vs=all_vp_vs[idx_ord],
                    poisson=all_pois[idx_ord],
                    x_twd97=float(all_x[i0]),
                    y_twd97=float(all_y[i0]),
                    dist_to_fault_km=float(all_fault[i0]) if all_fault is not None else None,
                    dist_to_coast_km=float(all_coast[i0]) if all_coast is not None else None,
                    include_sedimentary=include_sed
                )
                out_mat[idx_ord, :] = feats.values

            for c_i, col_name in enumerate(target_cols):
                res_df[col_name] = out_mat[:, c_i]
            res_df["_seq_enriched_v2"] = True
            return res_df
        else:
            df_sorted = df.sort_values(by="depth_m").copy() if len(df) > 1 and not df["depth_m"].is_monotonic_increasing else df.copy()
            feats = compute_column_sequence_features(
                depths=df_sorted["depth_m"].values,
                z_msl=df_sorted["z_msl"].values,
                vp=df_sorted["vp_ms"].values,
                vs=df_sorted["vs_ms"].values,
                vp_vs=df_sorted["vp_vs_ratio"].values,
                poisson=df_sorted["poisson_ratio"].values,
                x_twd97=float(df_sorted["x_twd97"].iloc[0]) if "x_twd97" in df_sorted.columns else 200000.0,
                y_twd97=float(df_sorted["y_twd97"].iloc[0]) if "y_twd97" in df_sorted.columns else 2600000.0,
                dist_to_fault_km=float(df_sorted["dist_to_fault_km"].iloc[0]) if "dist_to_fault_km" in df_sorted.columns else None,
                dist_to_coast_km=float(df_sorted["dist_to_coast_km"].iloc[0]) if "dist_to_coast_km" in df_sorted.columns else None,
                include_sedimentary=include_sed
            )
            for c in feats.columns:
                df_sorted[c] = feats[c].values
            df_sorted["_seq_enriched_v2"] = True
            return df_sorted

    def _fit_spatial_prior(self, train_df: pd.DataFrame):
        """僅在訓練集井位建立嚴格防洩漏之 2D KD-Tree 與深度分層岩相索引表。"""
        if "lithology_id" not in train_df.columns or "x_twd97" not in train_df.columns:
            self.has_spatial_prior = False
            return

        station_col = "station_id" if "station_id" in train_df.columns else ("station" if "station" in train_df.columns else None)
        if not station_col:
            self.has_spatial_prior = False
            return

        n_cls = len(self.l2_classes)
        df_valid = train_df[train_df["lithology_id"].isin(self.l2_to_idx.keys())]
        if len(df_valid) == 0:
            self.has_spatial_prior = False
            return

        # 1. 計算類別先驗分佈
        counts = np.ones(n_cls, dtype=np.float32)
        for c, idx in self.l2_to_idx.items():
            counts[idx] += float(np.sum(df_valid["lithology_id"].values == c))
        self.class_priors = counts / np.sum(counts)

        # 2. 建立每口訓練井在 0..350m (步長 2m, 共 176 格) 的岩相索引表 (純 NumPy 索引切片，避免數千次 pandas 子表複製)
        z_grid = np.arange(REF_N_DEPTH_BINS, dtype=float) * REF_DEPTH_STEP_M
        grouped_indices = df_valid.groupby(station_col, sort=False).indices
        n_wells = len(grouped_indices)

        all_x = df_valid["x_twd97"].values.astype(np.float32)
        all_y = df_valid["y_twd97"].values.astype(np.float32)
        all_d = df_valid["depth_m"].values.astype(float)
        all_l = np.array([self.l2_to_idx[c] for c in df_valid["lithology_id"].values], dtype=np.int8)

        ref_ids = []
        ref_xy = np.zeros((n_wells, 2), dtype=np.float32)
        ref_classes = np.full((n_wells, REF_N_DEPTH_BINS), -1, dtype=np.int8)
        global_counts = np.ones((REF_N_DEPTH_BINS, n_cls), dtype=np.float32) * 0.5

        for w_i, (st_id, idx_arr) in enumerate(grouped_indices.items()):
            ref_ids.append(str(st_id))
            ref_xy[w_i, 0] = all_x[idx_arr[0]]
            ref_xy[w_i, 1] = all_y[idx_arr[0]]

            d_vals = all_d[idx_arr]
            l_vals = all_l[idx_arr]
            if len(d_vals) > 1 and d_vals[0] > d_vals[-1]:
                order = np.argsort(d_vals)
                d_vals = d_vals[order]
                l_vals = l_vals[order]

            min_d = max(0.0, float(d_vals[0]) - 3.0)
            max_d = min(REF_MAX_DEPTH_M, float(d_vals[-1]) + 5.0)
            valid_mask = (z_grid >= min_d) & (z_grid <= max_d)
            if np.any(valid_mask):
                valid_z = z_grid[valid_mask]
                idx_right = np.clip(np.searchsorted(d_vals, valid_z), 0, len(d_vals) - 1)
                idx_left = np.clip(idx_right - 1, 0, len(d_vals) - 1)
                choose_left = np.abs(valid_z - d_vals[idx_left]) <= np.abs(d_vals[idx_right] - valid_z)
                nearest_idx = np.where(choose_left, idx_left, idx_right)
                assigned_cls = l_vals[nearest_idx]
                ref_classes[w_i, valid_mask] = assigned_cls
                for b_idx, c_idx in zip(np.where(valid_mask)[0], assigned_cls):
                    global_counts[b_idx, c_idx] += 1.0

        self.ref_station_ids = np.array(ref_ids, dtype=object)
        self.ref_xy = ref_xy
        self.ref_well_classes = ref_classes
        self.global_depth_prior = (global_counts / np.sum(global_counts, axis=1, keepdims=True)).astype(np.float32)
        self._kd_tree = cKDTree(self.ref_xy)
        # 建立深層觀測井 (>=45m, 即 bin>=22 具觀測值) 專屬子樹，避免淺孔密集叢集排擠深部鄰井檢索
        deep_mask = np.any(self.ref_well_classes[:, 22:] >= 0, axis=1)
        self.deep_well_indices = np.where(deep_mask)[0]
        self._kd_tree_deep = cKDTree(self.ref_xy[self.deep_well_indices]) if len(self.deep_well_indices) > 0 else None
        self.has_spatial_prior = True

    def compute_knn_prior_for_column(
        self,
        depths: np.ndarray,
        x_twd97: float,
        y_twd97: float,
        exclude_station_id: str | None = None,
        k_neighbors: int = 24,
        far_field_cutoff_m: float = 10000.0
    ) -> np.ndarray:
        """計算單井/單垂直柱之嚴格防洩漏近場 K-NN / IDW 岩相先驗。
        - 嚴格排除自身 station_id (Zero-Leakage)，防止訓練與評估時發生標籤自洩漏。
        - 雙尺度鄰井檢索：同時檢索最近之淺表鄰井與深部鄰井，防止淺層工程孔叢集排擠深部水文井先驗。
        - 在遠場 (>10km) 自動衰減退化為全域深度先驗 (alpha_near -> 0)。
        回傳形狀: (T, n_classes + 1)，前 n_classes 維為機率，最後 1 維為近場信心門控 alpha_near。
        """
        t_len = len(depths)
        n_cls = len(self.l2_classes)
        z_bins = np.clip(np.round(np.asarray(depths, dtype=float) / REF_DEPTH_STEP_M).astype(int), 0, REF_N_DEPTH_BINS - 1)

        if not self.has_spatial_prior or self._kd_tree is None or len(self.ref_xy) == 0:
            out = np.zeros((t_len, n_cls + 1), dtype=np.float32)
            out[:, :n_cls] = 1.0 / n_cls
            return out

        pt = [float(x_twd97), float(y_twd97)]
        k_query = min(k_neighbors + 4, len(self.ref_xy))
        dists, idxs = self._kd_tree.query(pt, k=k_query)
        if np.ndim(dists) == 0:
            dists = np.array([float(dists)])
            idxs = np.array([int(idxs)])

        # 若查詢柱深度超過 35m，同步自深井子樹檢索最近 16 口深井並聯集去重
        if len(depths) > 0 and float(np.max(depths)) > 35.0 and getattr(self, "_kd_tree_deep", None) is not None and len(self.deep_well_indices) > 0:
            k_deep = min(16, len(self.deep_well_indices))
            d_deep, i_deep_sub = self._kd_tree_deep.query(pt, k=k_deep)
            if np.ndim(d_deep) == 0:
                d_deep = np.array([float(d_deep)])
                i_deep_sub = np.array([int(i_deep_sub)])
            i_deep_global = self.deep_well_indices[i_deep_sub]
            combined_idxs = np.concatenate([idxs, i_deep_global])
            combined_dists = np.concatenate([dists, d_deep])
            _, uniq_pos = np.unique(combined_idxs, return_index=True)
            order_u = np.argsort(combined_dists[uniq_pos])
            idxs = combined_idxs[uniq_pos[order_u]]
            dists = combined_dists[uniq_pos[order_u]]

        # 嚴格排除自身 station_id (Zero-Leakage)
        if exclude_station_id is not None and self.ref_station_ids is not None:
            valid_nb = self.ref_station_ids[idxs] != str(exclude_station_id)
        else:
            valid_nb = np.ones(len(idxs), dtype=bool)

        # 同時过滤掉距離超過遠場截斷門檻 (>10km) 的鄰井
        valid_nb = valid_nb & (dists <= far_field_cutoff_m)
        dists = dists[valid_nb][:k_neighbors + 12]
        idxs = idxs[valid_nb][:k_neighbors + 12]

        global_p = self.global_depth_prior[z_bins]  # (T, n_cls)
        if len(idxs) == 0:
            out = np.zeros((t_len, n_cls + 1), dtype=np.float32)
            out[:, :n_cls] = global_p
            return out

        # 空間反距離權重 (IDW) + 指數距離衰減
        # 對同工區近距離鄰井 (10m~500m) 賦予強空間錨定權重，隨距離平滑衰減
        w_spatial = (1.0 / np.power(dists / 25.0 + 1.0, 2.25)) * np.exp(-dists / 2800.0)  # (K,)
        nb_classes = self.ref_well_classes[idxs]  # (K, 176)

        local_votes = np.zeros((t_len, n_cls), dtype=np.float32)
        min_valid_dist = np.full(t_len, far_field_cutoff_m + 1.0, dtype=np.float32)

        # 結合上下鄰近深度窗 (-2m, 0m, +2m) 兼顧薄互層解析度與微幅傾角穩健性
        z_kernel = [(-1, 0.22), (0, 0.56), (1, 0.22)]
        for dz_offset, w_z in z_kernel:
            b_idx = np.clip(z_bins + dz_offset, 0, REF_N_DEPTH_BINS - 1)
            cls_at_z = nb_classes[:, b_idx]  # (K, T)
            valid_at_z = cls_at_z >= 0       # (K, T)

            if abs(dz_offset) <= 1:
                for k_i in range(len(dists)):
                    m_k = valid_at_z[k_i]
                    min_valid_dist[m_k] = np.minimum(min_valid_dist[m_k], float(dists[k_i]))

            for c_i in range(n_cls):
                match_c = (cls_at_z == c_i) & valid_at_z  # (K, T)
                local_votes[:, c_i] += float(w_z) * np.dot(w_spatial.astype(np.float32), match_c.astype(np.float32))

        vote_sums = np.sum(local_votes, axis=1, keepdims=True)
        has_local = (vote_sums.squeeze(-1) > 1e-6) & (min_valid_dist <= far_field_cutoff_m)

        # 近場信心門控 alpha_near: 同工區 (<500m) 趨近 1.0，中場 (1~3km) 適度融合，遠場 (>10km) 歸零
        alpha_near = np.where(
            has_local,
            0.65 * np.exp(-min_valid_dist / 900.0) + 0.35 * np.exp(-np.square(min_valid_dist / 3200.0)),
            0.0
        ).astype(np.float32)[:, None]

        # 加入適度拉普拉斯平滑避免單一類別機率為 0 造成對數發散
        raw_local_probs = np.where(vote_sums > 1e-6, local_votes / np.maximum(vote_sums, 1e-6), global_p)
        local_probs = 0.92 * raw_local_probs + 0.08 * global_p
        blended_probs = alpha_near * local_probs + (1.0 - alpha_near) * global_p

        return np.concatenate([blended_probs.astype(np.float32), alpha_near], axis=-1)

    def fit(self, train_df: pd.DataFrame) -> SequenceFeatureExtractor:
        """僅在訓練集井位上擬合標準化尺度與近場空間錨定先驗。"""
        enriched = self.enrich_dataframe(train_df)
        X = enriched[self.feature_names].values
        self.scaler.fit(X)
        self._fit_spatial_prior(train_df)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """轉換特徵矩陣為正規化數值 (N, D)。"""
        if not self.fitted:
            raise RuntimeError("SequenceFeatureExtractor 尚未 fit，請先呼叫 fit(train_df)！")
        enriched = self.enrich_dataframe(df)
        X = enriched[self.feature_names].values
        return self.scaler.transform(X)

    def get_state(self) -> dict:
        state = {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "var": self.scaler.var_,
            "n_features_in": self.scaler.n_features_in_,
            "feature_names": self.feature_names,
            "l2_classes": self.l2_classes,
            "has_spatial_prior": self.has_spatial_prior
        }
        if self.has_spatial_prior:
            state["ref_station_ids"] = self.ref_station_ids
            state["ref_xy"] = self.ref_xy
            state["ref_well_classes"] = self.ref_well_classes
            state["global_depth_prior"] = self.global_depth_prior
            state["class_priors"] = self.class_priors
        return state

    def load_state(self, state: dict):
        self.scaler.mean_ = state["mean"]
        self.scaler.scale_ = state["scale"]
        self.scaler.var_ = state["var"]
        self.scaler.n_features_in_ = state["n_features_in"]
        self.feature_names = state.get("feature_names", list(FEATURE_NAMES))
        self.l2_classes = state.get("l2_classes", [2, 3, 4, 5, 6, 7])
        self.l2_to_idx = {c: i for i, c in enumerate(self.l2_classes)}
        self.has_spatial_prior = bool(state.get("has_spatial_prior", False))
        if self.has_spatial_prior and "ref_xy" in state and state["ref_xy"] is not None:
            self.ref_station_ids = state.get("ref_station_ids")
            self.ref_xy = np.asarray(state["ref_xy"], dtype=np.float32)
            self.ref_well_classes = np.asarray(state["ref_well_classes"], dtype=np.int8)
            self.global_depth_prior = np.asarray(state["global_depth_prior"], dtype=np.float32)
            self.class_priors = state.get("class_priors")
            self._kd_tree = cKDTree(self.ref_xy) if len(self.ref_xy) > 0 else None
            deep_mask = np.any(self.ref_well_classes[:, 22:] >= 0, axis=1)
            self.deep_well_indices = np.where(deep_mask)[0]
            self._kd_tree_deep = cKDTree(self.ref_xy[self.deep_well_indices]) if len(self.deep_well_indices) > 0 else None
        self.fitted = True
