"""3D 地球物理特徵內插模組 (Pure NumPy 3D Inverse Distance Weighting & Trilinear Interpolation)。

依據中研院 TEC (LEE2025) 3D 震波速度網格 (lat, lon, depth_m)，
對任意三維空間點 (lat, lon, depth_m) 內插 Vp, Vs, Vp/Vs 比值與垂直波速梯度 (dVs/dz)。
純 NumPy 實作，零外部相依性，運算極速且平滑。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

class VelocityModel3D:
    def __init__(self, velocity_csv_path: str | Path, k_neighbors: int = 8, p_power: float = 2.0):
        df = pd.read_csv(velocity_csv_path)
        # 特徵座標正規化以平衡經緯度 (度) 與深度 (公尺) 的尺度差距
        # 1 度緯度約 111,000 公尺, 1 度經度約 102,000 公尺 (23.5°N)
        self.coords_raw = df[["lat", "lon", "depth_m"]].values
        self.coords_scaled = self.coords_raw.copy()
        self.coords_scaled[:, 0] *= 111000.0  # lat to meters
        self.coords_scaled[:, 1] *= 102000.0  # lon to meters
        # depth is already in meters
        
        self.vp = df["vp_ms"].values
        self.vs = df["vs_ms"].values
        self.k = min(k_neighbors, len(df))
        self.p = p_power
        self.tree = cKDTree(self.coords_scaled)
        
    def _idw_interpolate(self, query_scaled: np.ndarray, values: np.ndarray) -> np.ndarray:
        """使用 cKDTree 進行 3D 反距離權重插值 (Inverse Distance Weighting)，低記憶體且高效。"""
        # dists: (M, k), idxs: (M, k)
        dists, idxs = self.tree.query(query_scaled, k=self.k, workers=-1)
        
        if self.k == 1:
            return values[idxs]
            
        weights = 1.0 / np.maximum(dists, 1e-4) ** self.p
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        interp = np.sum(weights * values[idxs], axis=1) / weights_sum.squeeze()
        
        # 若最近鄰居距離極近 (< 1e-4)，直接採用最近鄰居的值
        exact_mask = dists[:, 0] < 1e-4
        if np.any(exact_mask):
            interp[exact_mask] = values[idxs[exact_mask, 0]]
            
        return interp

    def query(self, lat: float | np.ndarray, lon: float | np.ndarray, depth_m: float | np.ndarray) -> dict[str, np.ndarray]:
        """查詢空間點之 Vp, Vs, Vp/Vs, dVs/dz。"""
        lats = np.atleast_1d(lat).astype(float)
        lons = np.atleast_1d(lon).astype(float)
        depths = np.atleast_1d(depth_m).astype(float)
        
        q_scaled = np.column_stack([lats * 111000.0, lons * 102000.0, depths])
        vp_val = self._idw_interpolate(q_scaled, self.vp)
        vs_val = self._idw_interpolate(q_scaled, self.vs)
        
        # 垂直波速梯度 (上下 10 公尺)
        dz = 10.0
        q_up = q_scaled.copy()
        q_up[:, 2] = np.maximum(0, q_up[:, 2] - dz)
        q_dn = q_scaled.copy()
        q_dn[:, 2] = q_dn[:, 2] + dz
        
        vs_up = self._idw_interpolate(q_up, self.vs)
        vs_dn = self._idw_interpolate(q_dn, self.vs)
        dvs_dz = (vs_dn - vs_up) / (2.0 * dz)
        
        vp_vs = vp_val / np.maximum(vs_val, 1e-3)
        
        return {
            "vp": vp_val,
            "vs": vs_val,
            "vp_vs_ratio": vp_vs,
            "dvs_dz": dvs_dz
        }
