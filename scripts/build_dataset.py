"""建構特徵資料集 (dataset_v1.parquet)。

整合：
1. 91 口深層水文觀測井之 24,767 筆標準化地層分層。
2. 測站高程與 TWD97 / WGS84 座標系統。
3. 中研院 TEC LEE2025 3D 震波速度模型內插 (Vp, Vs, Vp/Vs, dVs/dz)。
4. 海拔相對高程 (z_msl = elevation_m - depth_m)。

輸出：
  data/processed/dataset_v1.csv
  data/processed/dataset_v1.parquet
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.features.interpolator import VelocityModel3D
from tages.geo.crs import convert

WESTERN_DEFORMATION_FRONT = np.array([
    [121.50, 25.05],
    [121.25, 24.85],
    [120.95, 24.45],
    [120.72, 24.15],
    [120.58, 23.65],
    [120.48, 23.35],
    [120.42, 22.85],
    [120.45, 22.40],
    [120.75, 21.95]
])

WESTERN_COASTLINE = np.array([
    [121.50, 25.25],
    [121.15, 25.10],
    [120.95, 24.75],
    [120.62, 24.35],
    [120.30, 23.95],
    [120.12, 23.55],
    [120.08, 23.10],
    [120.20, 22.50],
    [120.65, 21.90]
])

def point_to_polyline_dist_km(lons: np.ndarray, lats: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """計算點集到多段線之最近水平距離 (km)。"""
    lat_mean = np.radians(np.mean(lats))
    scale_y = 111.0
    scale_x = 111.0 * np.cos(lat_mean)
    
    px = lons * scale_x
    py = lats * scale_y
    poly_x = polyline[:, 0] * scale_x
    poly_y = polyline[:, 1] * scale_y
    
    min_dists = np.full(len(lons), np.inf)
    for i in range(len(polyline) - 1):
        x1, y1 = poly_x[i], poly_y[i]
        x2, y2 = poly_x[i+1], poly_y[i+1]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx*dx + dy*dy
        if l2 == 0:
            d = np.hypot(px - x1, py - y1)
        else:
            t = np.clip(((px - x1)*dx + (py - y1)*dy) / l2, 0.0, 1.0)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = np.hypot(px - proj_x, py - proj_y)
        min_dists = np.minimum(min_dists, d)
    return np.round(min_dists, 2)

def build_dataset(
    boreholes_path: str = "data/interim/taiwan_boreholes.csv",
    strata_path: str = "data/interim/taiwan_strata.csv",
    velocity_grid_path: str = "data/interim/taiwan_velocity_3d.csv",
    out_dir: str = "data/processed",
    sample_step_m: float = 1.0  # 厚地層每隔多少公尺取樣一點
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("載入鑽孔與地層資料...")
    bh = pd.read_csv(boreholes_path)
    st = pd.read_csv(strata_path)
    
    # 合併鑽孔空間座標
    merged = pd.merge(st, bh[["station", "name", "city", "wsect", "lon", "lat", "x97", "y97", "elev_m", "well_depth_m"]], on="station", how="inner")
    print(f"有效分層紀錄: {len(merged)} 筆，來自 {merged.station.nunique()} 口深井")
    
    # 進行深度採樣：每層在中心點取樣，若厚度 > 2m 則等間距細分
    samples = []
    for _, row in merged.iterrows():
        top, bot = row["top_m"], row["bottom_m"]
        thick = max(0.1, bot - top)
        
        # 排除無效地層或標籤為 Unknown (0) 的管線/空白段
        if row.get("l2", row["l1"]) == 0:
            continue
            
        if thick <= sample_step_m:
            depths = [(top + bot) / 2.0]
        else:
            depths = list(np.arange(top + 0.5, bot, sample_step_m))
            if not depths:
                depths = [(top + bot) / 2.0]
                
        for d in depths:
            samples.append({
                "station_id": row["station"],
                "station_name": row["name"],
                "city": row["city"],
                "wsect": row["wsect"],
                "lon": row["lon"],
                "lat": row["lat"],
                "x_twd97": row["x97"],
                "y_twd97": row["y97"],
                "elevation_m": row["elev_m"],
                "depth_m": round(d, 2),
                "z_msl": round(row["elev_m"] - d, 2),
                "layer_top_m": top,
                "layer_bottom_m": bot,
                "layer_thickness_m": round(thick, 2),
                "rock_code": row["rock_code"],
                "lithology_l1_id": row["l1"],
                "lithology_l1_name": row["l1_name"],
                "lithology_id": row["l2"],
                "lithology_name": row["l2_name"]
            })
            
    df_samples = pd.DataFrame(samples)
    print(f"細分深度採樣完成！總特徵樣本數: {len(df_samples)} 筆")
    
    # 載入 3D 震波速度模型並批次內插
    print("正在內插 TEC 3D 震波速度特徵 (Vp, Vs, Vp/Vs, dVs/dz)...")
    vm = VelocityModel3D(velocity_grid_path)
    
    vel_feats = vm.query(
        lat=df_samples["lat"].values,
        lon=df_samples["lon"].values,
        depth_m=df_samples["depth_m"].values
    )
    
    df_samples["vp_ms"] = np.round(vel_feats["vp"], 1)
    df_samples["vs_ms"] = np.round(vel_feats["vs"], 1)
    df_samples["vp_vs_ratio"] = np.round(vel_feats["vp_vs_ratio"], 3)
    df_samples["dvs_dz"] = np.round(vel_feats["dvs_dz"], 4)
    
    # 衍生特徵：Poisson's Ratio (泊松比)
    # nu = 0.5 * ( (Vp/Vs)^2 - 2 ) / ( (Vp/Vs)^2 - 1 )
    r2 = df_samples["vp_vs_ratio"] ** 2
    df_samples["poisson_ratio"] = np.round(0.5 * (r2 - 2) / np.maximum(r2 - 1, 1e-4), 3)

    # 沉積地貌與壓密特徵
    print("正在計算構造前緣距離、海岸線距離與沉積壓密特徵...")
    df_samples["dist_to_fault_km"] = point_to_polyline_dist_km(
        df_samples["lon"].values, df_samples["lat"].values, WESTERN_DEFORMATION_FRONT
    )
    df_samples["dist_to_coast_km"] = point_to_polyline_dist_km(
        df_samples["lon"].values, df_samples["lat"].values, WESTERN_COASTLINE
    )
    compaction_vs = 160.0 * (1.0 + df_samples["depth_m"] / 8.0) ** 0.32
    df_samples["compaction_vs_ms"] = np.round(compaction_vs, 1)
    df_samples["vs_residual_ms"] = np.round(df_samples["vs_ms"] - compaction_vs, 1)
    
    # 存檔
    csv_out = out / "dataset_v1.csv"
    parquet_out = out / "dataset_v1.parquet"
    
    df_samples.to_csv(csv_out, index=False, encoding="utf-8-sig")
    try:
        df_samples.to_parquet(parquet_out, index=False)
        print(f"Parquet 儲存成功: {parquet_out}")
    except Exception as e:
        print(f"Parquet 儲存略過 (缺套件): {e}")
        
    print(f"CSV 儲存成功: {csv_out}")
    print("\n資料集摘要統計:")
    print(df_samples[["depth_m", "z_msl", "vp_ms", "vs_ms", "vp_vs_ratio", "poisson_ratio", "dist_to_fault_km", "dist_to_coast_km", "vs_residual_ms"]].describe().round(2).to_string())
    print("\n岩性類別分佈 (樣本數與百分比):")
    counts = df_samples["lithology_name"].value_counts()
    pcts = (counts / len(df_samples) * 100).round(1)
    print(pd.concat([counts.rename("Count"), pcts.rename("Pct%")], axis=1).to_string())
    
    return df_samples

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--boreholes", default="data/interim/taiwan_boreholes.csv", help="觀測井檔案")
    ap.add_argument("--strata", default="data/interim/taiwan_strata.csv", help="地層紀錄檔案")
    ap.add_argument("--velocity", default="data/interim/taiwan_velocity_3d.csv", help="3D 震波速度網格")
    ap.add_argument("--out", default="data/processed", help="輸出資料夾")
    ap.add_argument("--step", type=float, default=1.0, help="厚地層深度採樣步長(m)")
    args = ap.parse_args()

    build_dataset(
        boreholes_path=args.boreholes,
        strata_path=args.strata,
        velocity_grid_path=args.velocity,
        out_dir=args.out,
        sample_step_m=args.step
    )
