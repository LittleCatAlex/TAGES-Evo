"""地質剖面平滑後處理與地層連續性優化 (Geological Post-Processing & Layer Smoothing)。

原理：
天然沉積地層具備垂直與水平連續性。未加後處理之像素級預測常有椒鹽雜訊。
透過 2D 眾數濾波 (Majority / Mode Filtering) 與連續性平滑，
消除孤立奇異值，還原連續的砂層與泥層沉積層序結構。
"""
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.features.interpolator import VelocityModel3D
from tages.geo.crs import convert

LITH_COLORS = {
    1: ("#D2B48C", "Topsoil / Fill"),
    2: ("#8B4513", "Clay / Mud"),
    3: ("#CD853F", "Silt"),
    4: ("#FFD700", "Fine Sand"),
    5: ("#DAA520", "Coarse Sand"),
    6: ("#B8860B", "Sand-Mud Mixed"),
    7: ("#708090", "Gravel"),
    8: ("#4682B4", "Mudstone / Shale"),
    9: ("#2E8B57", "Sandstone"),
    10: ("#1C3144", "Crystalline Basement")
}

def mode_filter_2d(matrix: np.ndarray, size: int = 5) -> np.ndarray:
    """以二維滑動視窗眾數濾波 (Majority Vote) 平滑地層界線。"""
    h, w = matrix.shape
    pad = size // 2
    padded = np.pad(matrix, pad, mode='edge')
    smoothed = np.zeros_like(matrix)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+size, j:j+size].ravel()
            vals, counts = np.unique(window, return_counts=True)
            smoothed[i, j] = vals[np.argmax(counts)]
            
    return smoothed

def generate_smooth_profile(
    lat1: float = 23.38, lon1: float = 120.15,
    lat2: float = 23.50, lon2: float = 120.55,
    max_depth_m: float = 300.0,
    nx: int = 180,
    nz: int = 120,
    filter_size: int = 5,
    out_png: str = "reports/chiayi_ai_profile_smooth.png"
):
    print(f"產生高解析連續平滑地質剖面: 網格 {nx}x{nz}, 濾波尺寸: {filter_size}...")
    clf = joblib.load("models/tages_rf_augmented.pkl")
    vm = VelocityModel3D("data/interim/chiayi_velocity_3d.csv")
    
    lats = np.linspace(lat1, lat2, nx)
    lons = np.linspace(lon1, lon2, nx)
    depths = np.linspace(0, max_depth_m, nz)
    
    x97_1, y97_1 = convert(lon1, lat1, "wgs84", "twd97")
    x97_2, y97_2 = convert(lon2, lat2, "wgs84", "twd97")
    total_dist_km = np.sqrt((x97_2 - x97_1)**2 + (y97_2 - y97_1)**2) / 1000.0
    dist_km = np.linspace(0, total_dist_km, nx)
    
    grid_dist, grid_z = np.meshgrid(dist_km, depths)
    grid_lat, _ = np.meshgrid(lats, depths)
    grid_lon, _ = np.meshgrid(lons, depths)
    
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()
    flat_depth = grid_z.ravel()
    
    elev_est = 5.0 + (flat_lat - lat1) / (lat2 - lat1 + 1e-6) * 75.0
    flat_z_msl = elev_est - flat_depth
    
    flat_x97, flat_y97 = [], []
    for lo, la in zip(flat_lon, flat_lat):
        x, y = convert(lo, la, "wgs84", "twd97")
        flat_x97.append(x)
        flat_y97.append(y)
    flat_x97 = np.array(flat_x97)
    flat_y97 = np.array(flat_y97)
    
    v_res = vm.query(flat_lat, flat_lon, flat_depth)
    vp = v_res["vp"]
    vs = v_res["vs"]
    vp_vs = v_res["vp_vs_ratio"]
    dvs_dz = v_res["dvs_dz"]
    r2 = vp_vs ** 2
    poisson = 0.5 * (r2 - 2) / np.maximum(r2 - 1, 1e-4)
    
    X_grid = np.column_stack([
        flat_depth, flat_z_msl, vp, vs, vp_vs, dvs_dz, poisson, flat_x97, flat_y97
    ])
    
    raw_preds = clf.predict(X_grid).reshape(nz, nx)
    probs = clf.predict_proba(X_grid)
    confidence = np.max(probs, axis=1).reshape(nz, nx)
    
    # 應用 2D 眾數濾波後處理
    smooth_preds = mode_filter_2d(raw_preds, size=filter_size)
    
    # 繪製對比圖
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
    
    def render_lith_image(pred_mat):
        c_mat = np.zeros((nz, nx, 3))
        for lith_id, (hex_c, _) in LITH_COLORS.items():
            rgb = [int(hex_c[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
            c_mat[pred_mat == lith_id] = rgb
        return c_mat
        
    # 1. 原始 AI 預測剖面
    ax1.imshow(render_lith_image(raw_preds), aspect='auto', extent=[0, total_dist_km, max_depth_m, 0])
    ax1.set_ylabel("Depth (m)", fontsize=11)
    ax1.set_title("A) Raw AI Pixel-wise Prediction Profile (Before Smoothing)", fontsize=13, fontweight='bold', loc='left')
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # 2. 地層連續性平滑後處理剖面
    ax2.imshow(render_lith_image(smooth_preds), aspect='auto', extent=[0, total_dist_km, max_depth_m, 0])
    ax2.set_ylabel("Depth (m)", fontsize=11)
    ax2.set_title("B) Geologically Consistent Continuous Strata Profile (Post-Processed)", fontsize=13, fontweight='bold', loc='left')
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    handles = [plt.Rectangle((0,0),1,1, color=hex_c) for lid, (hex_c, label) in LITH_COLORS.items() if lid in [2, 3, 4, 5, 6, 7]]
    labels = [LITH_COLORS[lid][1] for lid in [2, 3, 4, 5, 6, 7]]
    ax2.legend(handles, labels, loc="lower right", framealpha=0.9, fontsize=10, ncol=2)
    
    # 3. 信心度分佈
    im_conf = ax3.imshow(confidence, aspect='auto', extent=[0, total_dist_km, max_depth_m, 0], cmap="viridis", vmin=0.45, vmax=0.95)
    ax3.set_xlabel("Distance along profile (km)  [West -> East]", fontsize=11)
    ax3.set_ylabel("Depth (m)", fontsize=11)
    ax3.set_title("C) AI Prediction Confidence (Certainty Heatmap)", fontsize=12, loc='left')
    ax3.grid(True, linestyle="--", alpha=0.4)
    
    cbar = fig.colorbar(im_conf, ax=ax3, orientation='horizontal', pad=0.25, shrink=0.6)
    cbar.set_label("Confidence (Maximum Class Probability)", fontsize=10)
    
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"平滑對比剖面圖已儲存至: {out_png}")

if __name__ == "__main__":
    generate_smooth_profile()
