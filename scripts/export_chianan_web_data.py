"""匯出西南臺灣 (嘉南平原與雲嘉南地區) 3D 互動網頁資料模組。

自 chianan_3d_voxels_1km.npz 與 hydro_boreholes.csv 萃取精簡立體幾何數據與地質構造線，
產出 reports/chianan_3d_data.js (以 window.TAGES_3D_DATA 格式封裝，完全免除瀏覽器本機 CORS 限制)。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def export_web_data():
    root = Path(__file__).resolve().parents[1]
    npz_path = root / "data" / "processed" / "chianan_3d_voxels_1km.npz"
    borehole_path = root / "data" / "interim" / "hydro_boreholes.csv"
    out_js = root / "reports" / "chianan_3d_data.js"
    out_js.parent.mkdir(parents=True, exist_ok=True)

    print("載入西南臺灣 3D 體素資料庫...")
    data = np.load(npz_path)
    lats = data["lats"]          # (46,)
    lons = data["lons"]          # (46,)
    depths = data["depths"]      # (101,)
    lith_3d = data["lithology_3d"]  # (46, 46, 101)
    vs_3d = data["vs_3d"]
    conf_3d = data["confidence_3d"]
    bedrock_2d = data["bedrock_depth_2d"]  # (46, 46)

    # 1. 萃取 3D 點雲 (做適度空間降採樣以保證 WebGL 60FPS 流暢度: 經度 step=2, 緯度 step=2, 深度 step=2)
    # 產出約 23 * 23 * 51 = 26,979 點
    points = []
    step_lon = 2
    step_lat = 2
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

    # 2. 萃取真實水文地質深井 (涵蓋雲林、嘉義、臺南全區 91 口觀測井)
    df_bh = pd.read_csv(borehole_path)
    boreholes = []
    for _, row in df_bh.iterrows():
        boreholes.append({
            "id": str(row["station"]),
            "name": str(row["name"]),
            "city": str(row["city"]),
            "wsect": str(row.get("wsect", "")),
            "lon": round(float(row["lon"]), 4),
            "lat": round(float(row["lat"]), 4),
            "depth": round(float(row["well_depth_m"]), 1),
            "elev": round(float(row["elev_m"]), 1)
        })

    # 3. 萃取基盤網格曲面 (46 x 46)
    bedrock_grid = []
    for i in range(len(lats)):
        row = [round(float(bedrock_2d[i, j]), 1) for j in range(len(lons))]
        bedrock_grid.append(row)

    # 4. 主要活動斷層地表跡線 (Active Fault Traces)
    # 經緯度座標折線
    faults = [
        {
            "name": "Chukou Fault (觸口斷層)",
            "coords": [
                [120.59, 23.58], [120.57, 23.52], [120.56, 23.46],
                [120.55, 23.40], [120.54, 23.34], [120.53, 23.28]
            ],
            "color": "#ff3333"
        },
        {
            "name": "Meishan Fault (梅山斷層)",
            "coords": [
                [120.60, 23.58], [120.54, 23.57], [120.48, 23.56],
                [120.43, 23.55], [120.38, 23.55]
            ],
            "color": "#ff9900"
        },
        {
            "name": "Hsinhua Fault (新化斷層)",
            "coords": [
                [120.36, 23.03], [120.32, 23.03], [120.27, 23.04], [120.22, 23.04]
            ],
            "color": "#ff00ff"
        }
    ]

    payload = {
        "region": "Southwestern Taiwan (Yunlin-Chiayi-Tainan)",
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
        "bedrock_grid": bedrock_grid,
        "faults": faults
    }

    print(f"點雲總數: {len(points)} 點，真實水文觀測井: {len(boreholes)} 口，活動斷層: {len(faults)} 條")
    json_str = json.dumps(payload, ensure_ascii=False)
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.TAGES_3D_DATA = " + json_str + ";\n")

    size_mb = out_js.stat().st_size / 1024 / 1024
    print(f"3D 網頁資料匯出完成: {out_js} (體積: {size_mb:.2f} MB)")

if __name__ == "__main__":
    export_web_data()
