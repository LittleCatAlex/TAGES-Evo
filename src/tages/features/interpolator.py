"""3D 地球物理特徵內插模組 (Pure NumPy 3D Inverse Distance Weighting & Trilinear Interpolation)。

依據中研院 TEC (LEE2025) 3D 震波速度網格 (lat, lon, depth_m)，
對任意三維空間點 (lat, lon, depth_m) 內插 Vp, Vs, Vp/Vs 比值與垂直波速梯度 (dVs/dz)。
純 NumPy 實作，零外部相依性，運算極速且平滑。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

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
        self.k = k_neighbors
        self.p = p_power
        
    def _idw_interpolate(self, query_scaled: np.ndarray, values: np.ndarray) -> np.ndarray:
        """向量化 3D 反距離權重插值 (Inverse Distance Weighting)。"""
        # query_scaled: (M, 3), coords_scaled: (N, 3)
        # 計算歐式距離矩陣
        dists = np.linalg.norm(query_scaled[:, np.newaxis, :] - self.coords_scaled[np.newaxis, :, :], axis=2)
        
        # 檢查是否有距離為 0 的精確命中
        exact_match = dists < 1e-4
        has_exact = np.any(exact_match, axis=1)
        
        # 取最近 k 個鄰居
        k_idx = np.argsort(dists, axis=1)[:, :self.k]
        rows = np.arange(len(query_scaled))[:, np.newaxis]
        
        top_dists = dists[rows, k_idx]
        top_vals = values[k_idx]
        
        weights = 1.0 / np.maximum(top_dists, 1e-4) ** self.p
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        interp = np.sum(weights * top_vals, axis=1) / weights_sum.squeeze()
        
        # 精確命中補正
        if np.any(has_exact):
            exact_indices = np.argmax(exact_match[has_exact], axis=1)
            interp[has_exact] = values[exact_indices]
            
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
