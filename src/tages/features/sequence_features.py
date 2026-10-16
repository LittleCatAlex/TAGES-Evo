"""井身深度序列特徵計算與空間正規化模組 (Wellbore Sequence Feature Engineering)。

計算每口井沿垂直深度軸演進的 14 維物理特徵：
1. depth_m: 深度 (m)
2. z_msl: 海拔高程 (m)
3. rel_depth_ratio: 相對井深比例 (0 ~ 1.0)
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
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
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

def compute_column_sequence_features(
    depths: np.ndarray,
    z_msl: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    vp_vs: np.ndarray,
    poisson: np.ndarray,
    x_twd97: float,
    y_twd97: float,
    max_depth: float | None = None
) -> pd.DataFrame:
    """針對單一垂直柱/單一井位計算沿深度軸演進的 14 維特徵 DataFrame。"""
    n = len(depths)
    max_d = max_depth if (max_depth and max_depth > 0) else max(1.0, float(np.max(depths)))
    rel_depth = np.clip(depths / max_d, 0.0, 1.5)

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
    return pd.DataFrame(data)[FEATURE_NAMES]


class SequenceFeatureExtractor:
    """序列特徵標準化提取器，支援 fit(train_df) 與 transform(df)。"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """依井位 group 計算各井的時序衍生特徵並附加回 DataFrame。"""
        out_dfs = []
        station_col = "station_id" if "station_id" in df.columns else ("station" if "station" in df.columns else None)
        
        if station_col:
            grouped = df.groupby(station_col, sort=False)
            for _, grp in grouped:
                grp_sorted = grp.sort_values(by="depth_m").copy()
                feats = compute_column_sequence_features(
                    depths=grp_sorted["depth_m"].values,
                    z_msl=grp_sorted["z_msl"].values,
                    vp=grp_sorted["vp_ms"].values,
                    vs=grp_sorted["vs_ms"].values,
                    vp_vs=grp_sorted["vp_vs_ratio"].values,
                    poisson=grp_sorted["poisson_ratio"].values,
                    x_twd97=float(grp_sorted["x_twd97"].iloc[0]),
                    y_twd97=float(grp_sorted["y_twd97"].iloc[0]),
                    max_depth=float(grp_sorted["depth_m"].max())
                )
                for c in FEATURE_NAMES:
                    grp_sorted[c] = feats[c].values
                out_dfs.append(grp_sorted)
            return pd.concat(out_dfs, ignore_index=True)
        else:
            # 單一垂直序列
            feats = compute_column_sequence_features(
                depths=df["depth_m"].values,
                z_msl=df["z_msl"].values,
                vp=df["vp_ms"].values,
                vs=df["vs_ms"].values,
                vp_vs=df["vp_vs_ratio"].values,
                poisson=df["poisson_ratio"].values,
                x_twd97=float(df.get("x_twd97", pd.Series([200000.0])).iloc[0]),
                y_twd97=float(df.get("y_twd97", pd.Series([2600000.0])).iloc[0])
            )
            df_out = df.copy()
            for c in FEATURE_NAMES:
                df_out[c] = feats[c].values
            return df_out

    def fit(self, train_df: pd.DataFrame) -> SequenceFeatureExtractor:
        """僅在訓練集井位上擬合標準化尺度。"""
        enriched = self.enrich_dataframe(train_df)
        X = enriched[FEATURE_NAMES].values
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """轉換特徵矩陣為正規化數值 (N, 14)。"""
        if not self.fitted:
            raise RuntimeError("SequenceFeatureExtractor 尚未 fit，請先呼叫 fit(train_df)！")
        enriched = self.enrich_dataframe(df)
        X = enriched[FEATURE_NAMES].values
        return self.scaler.transform(X)

    def get_state(self) -> dict:
        return {
            "mean": self.scaler.mean_,
            "scale": self.scaler.scale_,
            "var": self.scaler.var_,
            "n_features_in": self.scaler.n_features_in_,
            "feature_names": FEATURE_NAMES
        }

    def load_state(self, state: dict):
        self.scaler.mean_ = state["mean"]
        self.scaler.scale_ = state["scale"]
        self.scaler.var_ = state["var"]
        self.scaler.n_features_in_ = state["n_features_in"]
        self.fitted = True
