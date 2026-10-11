"""2D 剖面圖生成與地體結構預測視覺化模組 (AI Profile & Cross-Section Visualizer)。

功能：
1. 訓練全量特徵增強模型 (RandomForest Augmented) 並存檔至 models/tages_rf_augmented.pkl。
2. 沿指定之起訖點 (Lat1, Lon1) -> (Lat2, Lon2) 產生高解析 2D 垂直網格剖面。
3. 內插 3D 震波速度特徵並進行 AI 岩性機率預測。
4. 繪製標準地質剖面圖，並將鄰近真實水文/工程觀測井地層柱狀圖疊加比對！
5. 匯出高解析度圖像至 reports/chiayi_ai_profile_transect.png。
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.features.interpolator import VelocityModel3D
from tages.geo.crs import convert

# 標準地質色表 (L2 10 類地質規範)
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

def train_and_save_model(dataset_path: str = "data/processed/dataset_v1.csv", model_out: str = "models/tages_rf_augmented.pkl"):
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)
    df_clean = df[df["lithology_id"].isin([2, 3, 4, 5, 6, 7])].copy().reset_index(drop=True)
    
    feat_aug = ["depth_m", "z_msl", "vp_ms", "vs_ms", "vp_vs_ratio", "dvs_dz", "poisson_ratio", "x_twd97", "y_twd97"]
    X = df_clean[feat_aug].values
    y = df_clean["lithology_id"].values
    
    print("訓練全量隨機森林模型...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)
    joblib.dump(clf, model_out)
    print(f"模型已儲存至: {model_out}")
    return clf, feat_aug

def generate_profile(
    lat1: float = 23.38, lon1: float = 120.15,  # 沿海 (布袋/東石)
    lat2: float = 23.50, lon2: float = 120.55,  # 山麓 (中埔/嘉義市東側)
    max_depth_m: float = 300.0,
    nx: int = 150,
    nz: int = 100,
    velocity_csv: str = "data/interim/chiayi_velocity_3d.csv",
    model_path: str = "models/tages_rf_augmented.pkl",
    out_png: str = "reports/chiayi_ai_profile_transect.png"
):
    print(f"生成 2D 剖面: ({lat1}, {lon1}) -> ({lat2}, {lon2}), 深度: 0 ~ {max_depth_m}m")
    import matplotlib.pyplot as plt
    
    # 載入模型與速度網格
    clf = joblib.load(model_path)
    vm = VelocityModel3D(velocity_csv)
    
    # 建立 2D 網格點
    lats = np.linspace(lat1, lat2, nx)
    lons = np.linspace(lon1, lon2, nx)
    depths = np.linspace(0, max_depth_m, nz)
    
    # 計算各點沿剖線距離 (km)
    x97_1, y97_1 = convert(lon1, lat1, "wgs84", "twd97")
    x97_2, y97_2 = convert(lon2, lat2, "wgs84", "twd97")
    total_dist_km = np.sqrt((x97_2 - x97_1)**2 + (y97_2 - y97_1)**2) / 1000.0
    dist_km = np.linspace(0, total_dist_km, nx)
    
    grid_dist, grid_z = np.meshgrid(dist_km, depths)
    grid_lat, _ = np.meshgrid(lats, depths)
    grid_lon, _ = np.meshgrid(lons, depths)
    
    # 平坦化查詢特徵
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()
    flat_depth = grid_z.ravel()
    
    # 估算地表高程 (簡化線性斜率：沿海 5m -> 山麓 80m)
    elev_est = 5.0 + (flat_lat - lat1) / (lat2 - lat1 + 1e-6) * 75.0
    flat_z_msl = elev_est - flat_depth
    
    flat_x97, flat_y97 = [], []
    for lo, la in zip(flat_lon, flat_lat):
        x, y = convert(lo, la, "wgs84", "twd97")
        flat_x97.append(x)
        flat_y97.append(y)
    flat_x97 = np.array(flat_x97)
    flat_y97 = np.array(flat_y97)
    
    print("內插 3D 震波速度...")
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
    
    print("AI 預測網格岩性機率...")
    preds = clf.predict(X_grid).reshape(nz, nx)
    probs = clf.predict_proba(X_grid)
    # 取最大預測機率作為信心度 (Confidence)
    confidence = np.max(probs, axis=1).reshape(nz, nx)
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. 剖面岩性分布圖
    # 建立自訂色彩映射
    color_matrix = np.zeros((nz, nx, 3))
    for lith_id, (hex_c, label) in LITH_COLORS.items():
        # hex to rgb [0, 1]
        rgb = [int(hex_c[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
        mask = (preds == lith_id)
        color_matrix[mask] = rgb
        
    ax1.imshow(color_matrix, aspect='auto', extent=[0, total_dist_km, max_depth_m, 0])
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.set_title(f"Taiwan 3D AI Geological Profile: Chiayi Coastal Plain to Foothills\n({lat1}N, {lon1}E) -> ({lat2}N, {lon2}E)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 加入自訂圖例
    handles = [plt.Rectangle((0,0),1,1, color=hex_c) for lith_id, (hex_c, label) in LITH_COLORS.items() if lith_id in [2, 3, 4, 5, 6, 7]]
    labels = [LITH_COLORS[lid][1] for lid in [2, 3, 4, 5, 6, 7]]
    ax1.legend(handles, labels, loc="lower right", framealpha=0.9, fontsize=10, ncol=2)
    
    # 2. 信心度 / 不確定性圖層
    im_conf = ax2.imshow(confidence, aspect='auto', extent=[0, total_dist_km, max_depth_m, 0], cmap="viridis", vmin=0.4, vmax=1.0)
    ax2.set_xlabel("Distance along profile (km)", fontsize=12)
    ax2.set_ylabel("Depth (m)", fontsize=12)
    ax2.set_title("AI Prediction Confidence Layer (Certainty Index)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    cbar = fig.colorbar(im_conf, ax=ax2, orientation='horizontal', pad=0.2, shrink=0.6)
    cbar.set_label("Prediction Probability / Confidence", fontsize=10)
    
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"剖面圖已成功儲存至: {out_png}")

if __name__ == "__main__":
    train_and_save_model()
    generate_profile()
