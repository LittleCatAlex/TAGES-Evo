"""匯出 3D 互動網頁資料模組 (Export 3D Data for Web Visualizer)。

自 chiayi_3d_voxels_1km.npz 與 hydro_boreholes.csv 萃取精簡立體幾何數據，
產出 reports/chiayi_3d_data.js (以 window.TAGES_3D_DATA 格式封裝，完全免除瀏覽器本機 CORS 限制)。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def export_web_data():
    root = Path(__file__).resolve().parents[1]
    npz_path = root / "data" / "processed" / "chiayi_3d_voxels_1km.npz"
    borehole_path = root / "data" / "interim" / "hydro_boreholes.csv"
    out_js = root / "reports" / "chiayi_3d_data.js"
    out_js.parent.mkdir(parents=True, exist_ok=True)

    print("載入 3D 體素資料庫...")
    data = np.load(npz_path)
    lats = data["lats"]          # (26,)
    lons = data["lons"]          # (51,)
    depths = data["depths"]      # (101,)
    lith_3d = data["lithology_3d"]  # (26, 51, 101)
    vs_3d = data["vs_3d"]
    conf_3d = data["confidence_3d"]
    bedrock_2d = data["bedrock_depth_2d"]  # (26, 51)

    # 1. 萃取 3D 點雲 (做適度空間降採樣以保證 WebGL 60FPS 流暢度: 經度取 step=2, 緯度取 step=1, 深度取 step=2)
    # 產出約 26 * 26 * 51 = 34,476 點
    points = []
    step_lon = 2
    step_lat = 1
    step_depth = 2

    for i in range(0, len(lats), step_lat):
        lat = float(lats[i])
        for j in range(0, len(lons), step_lon):
            lon = float(lons[j])
            for k in range(0, len(depths), step_depth):
                depth = float(depths[k])
                lid = int(lith_3d[i, j, k])
                vs = float(vs_3d[i, j, k])
                conf = float(conf_3d[i, j, k])
                # [lon, lat, depth, lid, vs, conf]
                points.append([
                    round(lon, 4),
                    round(lat, 4),
                    round(depth, 1),
                    lid,
                    round(vs, 0),
                    round(conf, 2)
                ])

    # 2. 萃取真實水文地質深井 (篩選位於嘉義範圍內之觀測井)
    df_bh = pd.read_csv(borehole_path)
    mask_chiayi = (
        (df_bh["lon"] >= lons.min() - 0.05) & (df_bh["lon"] <= lons.max() + 0.05) &
        (df_bh["lat"] >= lats.min() - 0.05) & (df_bh["lat"] <= lats.max() + 0.05)
    )
    df_cy = df_bh[mask_chiayi].copy()
    boreholes = []
    for _, row in df_cy.iterrows():
        boreholes.append({
            "id": str(row["station"]),
            "name": str(row["name"]),
            "city": str(row["city"]),
            "lon": round(float(row["lon"]), 4),
            "lat": round(float(row["lat"]), 4),
            "depth": round(float(row["well_depth_m"]), 1),
            "elev": round(float(row["elev_m"]), 1)
        })

    # 3. 萃取基盤網格曲面
    bedrock_grid = []
    for i in range(len(lats)):
        row = [round(float(bedrock_2d[i, j]), 1) for j in range(len(lons))]
        bedrock_grid.append(row)

    payload = {
        "bounds": {
            "lon_min": float(lons.min()),
            "lon_max": float(lons.max()),
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "depth_min": float(depths.min()),
            "depth_max": float(depths.max())
        },
        "lats": [round(float(x), 4) for x in lats],
        "lons": [round(float(x), 4) for x in lons],
        "depths": [round(float(x), 1) for x in depths],
        "points": points,
        "boreholes": boreholes,
        "bedrock_grid": bedrock_grid
    }

    print(f"點雲總數: {len(points)} 點，真實觀測井數: {len(boreholes)} 口")
    json_str = json.dumps(payload, ensure_ascii=False)
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.TAGES_3D_DATA = " + json_str + ";\n")

    size_mb = out_js.stat().st_size / 1024 / 1024
    print(f"3D 網頁資料匯出完成: {out_js} (體積: {size_mb:.2f} MB)")

if __name__ == "__main__":
    export_web_data()
