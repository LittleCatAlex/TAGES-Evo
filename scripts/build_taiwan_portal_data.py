"""建構臺灣全島 2D GIS 地圖門戶與高密度 3D 立體地體檢視器數據模組。

產出 reports/taiwan_portal_data.js，包含：
1. 全台 423 口深層水文地質觀測井基本資料與座標。
2. 全台 9 大主要活動斷層地表跡線與三維走向。
3. 全台六大代表性盆地/平原之高密度 3D 體素數據包 (點雲、基盤曲面、空間界限)。
4. 支援在 2D 地圖上任意拉框/點選目標範圍 (ROI)，切換至 3D 視窗檢視！
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    root = Path(__file__).resolve().parents[1]
    bh_csv = root / "data" / "interim" / "taiwan_boreholes.csv"
    school_bh_csv = root / "data" / "interim" / "school_boreholes.csv"
    npz_path = root / "data" / "processed" / "taiwan_basins_dense.npz"
    out_js = root / "reports" / "taiwan_portal_data.js"
    out_js.parent.mkdir(parents=True, exist_ok=True)

    print("載入全台水文觀測井與工程地質鑽孔資料庫...")
    df_bh = pd.read_csv(bh_csv)
    if school_bh_csv.exists():
        df_sch = pd.read_csv(school_bh_csv)
        df_bh = pd.concat([df_bh, df_sch], ignore_index=True).drop_duplicates(subset=["station"])
    boreholes = []
    for _, r in df_bh.iterrows():
        boreholes.append({
            "id": str(r["station"]),
            "name": str(r["name"]),
            "city": str(r["city"]),
            "wsect": str(r.get("wsect", "")),
            "lon": round(float(r["lon"]), 4),
            "lat": round(float(r["lat"]), 4),
            "depth": round(float(r["well_depth_m"]), 1),
            "elev": round(float(r["elev_m"]), 1),
            "year": int(r.get("year", 0)),
            "has_phys": bool(r.get("has_phys", False)),
            "has_c14": bool(r.get("has_c14", False))
        })
    print(f"  已收錄 {len(boreholes)} 口全台觀測井與工程鑽孔")

    # 全台九大主要活動斷層地表跡線
    faults = [
        {
            "name": "Chelungpu Fault (車籠埔斷層 - 921 地震地表破裂帶)",
            "region": "中部 (臺中/南投)",
            "color": "#ff2222",
            "coords": [
                [120.73, 24.32], [120.73, 24.26], [120.72, 24.20],
                [120.71, 24.13], [120.70, 24.04], [120.69, 23.94], [120.68, 23.82]
            ]
        },
        {
            "name": "Chukou Fault (觸口斷層 - 嘉南平原與麓山帶交界)",
            "region": "南部 (嘉義/臺南)",
            "color": "#ff4444",
            "coords": [
                [120.59, 23.58], [120.57, 23.52], [120.56, 23.46],
                [120.55, 23.40], [120.54, 23.34], [120.53, 23.28]
            ]
        },
        {
            "name": "Meishan Fault (梅山斷層 - 1906 嘉義大地震破裂帶)",
            "region": "南部 (嘉義)",
            "color": "#ff8800",
            "coords": [
                [120.60, 23.58], [120.54, 23.57], [120.48, 23.56],
                [120.43, 23.55], [120.38, 23.55]
            ]
        },
        {
            "name": "Shanchiao Fault (山腳斷層 - 臺北半地塹正斷層)",
            "region": "北部 (臺北/新北)",
            "color": "#ff0077",
            "coords": [
                [121.41, 25.04], [121.43, 25.08], [121.47, 25.13],
                [121.52, 25.18], [121.58, 25.24]
            ]
        },
        {
            "name": "Sanyi Fault (三義斷層)",
            "region": "苗栗/臺中",
            "color": "#ffaa00",
            "coords": [
                [120.76, 24.42], [120.74, 24.36], [120.72, 24.30]
            ]
        },
        {
            "name": "Dajia Fault (大甲斷層)",
            "region": "臺中沿海",
            "color": "#ff6600",
            "coords": [
                [120.61, 24.38], [120.58, 24.30], [120.56, 24.22]
            ]
        },
        {
            "name": "Hsinhua Fault (新化斷層 - 1946 臺南大地震破裂帶)",
            "region": "臺南",
            "color": "#cc00ff",
            "coords": [
                [120.36, 23.03], [120.32, 23.03], [120.27, 23.04], [120.22, 23.04]
            ]
        },
        {
            "name": "Chaochou Fault (潮州斷層 - 屏東平原東界深大斷裂)",
            "region": "屏東",
            "color": "#e60000",
            "coords": [
                [120.63, 22.82], [120.62, 22.72], [120.61, 22.61],
                [120.60, 22.50], [120.61, 22.40]
            ]
        },
        {
            "name": "Chihshang Fault (池上斷層 - 板塊縫合帶潛移斷層)",
            "region": "花蓮/臺東",
            "color": "#00ddff",
            "coords": [
                [121.32, 23.28], [121.27, 23.18], [121.22, 23.08],
                [121.18, 22.98], [121.14, 22.88]
            ]
        }
    ]

    # 全台六大代表性盆地配置
    basins_meta = {
        "chianan": {
            "name": "嘉南平原 (Chianan Plain)",
            "city": "雲林・嘉義・臺南",
            "desc": "臺灣最大前陸沖積平原，北港基底高區與臺南深前陸槽之典型對比地體",
            "center": [120.34, 23.40],
            "zoom": 10,
            "bounds": [120.08, 23.15, 120.60, 23.65]
        },
        "zhuoshui": {
            "name": "濁水溪沖積扇 (Zhuoshui Fan)",
            "city": "彰化・雲林",
            "desc": "臺灣最大現代沖積扇，扇頂巨厚礫石層至扇端細粒砂泥層序",
            "center": [120.40, 23.77],
            "zoom": 10,
            "bounds": [120.15, 23.65, 120.65, 23.90]
        },
        "taipei": {
            "name": "臺北盆地 (Taipei Basin)",
            "city": "臺北市・新北市",
            "desc": "半地塹構造陷落盆地，山腳正斷層下盤，覆蓋巨厚松山層泥砂與景美礫石層",
            "center": [121.52, 25.05],
            "zoom": 11,
            "bounds": [121.42, 24.98, 121.62, 25.12]
        },
        "taichung": {
            "name": "臺中盆地 (Taichung Basin)",
            "city": "臺中市・南投縣",
            "desc": "車籠埔逆衝斷層下盤盆地，頭嵙山層巨厚礫岩與大肚山台地交會",
            "center": [120.68, 24.18],
            "zoom": 11,
            "bounds": [120.58, 24.08, 120.78, 24.28]
        },
        "pingtung": {
            "name": "屏東平原 (Pingtung Plain)",
            "city": "屏東縣・高雄市",
            "desc": "潮州斷層以西厚層砂礫石前陸槽，高屏溪大型複式沖積扇構造",
            "center": [120.55, 22.63],
            "zoom": 10,
            "bounds": [120.42, 22.48, 120.68, 22.78]
        },
        "yilan": {
            "name": "蘭陽平原 (Lanyang Plain)",
            "city": "宜蘭縣",
            "desc": "沖繩海槽西延弧後張裂三角洲，扇頂厚層粗礫石至海濱細泥",
            "center": [121.78, 24.75],
            "zoom": 11,
            "bounds": [121.68, 24.65, 121.88, 24.85]
        }
    }

    # 載入預先計算之六大盆地高密度 3D 體素數據
    print("載入六大盆地高密度 3D 體素數據包...")
    npz_data = np.load(npz_path)

    basins_voxels = {}
    for b_key in basins_meta.keys():
        lats = npz_data[f"{b_key}_lats"]
        lons = npz_data[f"{b_key}_lons"]
        depths = npz_data[f"{b_key}_depths"]
        lith_3d = npz_data[f"{b_key}_lith"]
        vs_3d = npz_data[f"{b_key}_vs"]
        conf_3d = npz_data[f"{b_key}_conf"]
        bedrock_2d = npz_data[f"{b_key}_bedrock"]

        # 整理點雲 (步長取 1 以保留高密度緻密實體體素！)
        pts = []
        n_lat, n_lon, n_depth = lith_3d.shape
        for i in range(n_lat):
            la = float(lats[i])
            for j in range(n_lon):
                lo = float(lons[j])
                for k in range(n_depth):
                    d = float(depths[k])
                    lid = int(lith_3d[i, j, k])
                    vs = float(vs_3d[i, j, k])
                    cf = float(conf_3d[i, j, k])
                    pts.append([
                        round(lo, 4), round(la, 4), round(d, 1),
                        lid, round(vs, 0), round(cf, 2)
                    ])

        bedrock_grid = []
        bedrock_eng_grid = []
        bedrock_ac_grid = []
        has_eng = f"{b_key}_bedrock_eng" in npz_data
        has_ac = f"{b_key}_bedrock_ac" in npz_data
        bedrock_eng_2d = npz_data[f"{b_key}_bedrock_eng"] if has_eng else bedrock_2d
        bedrock_ac_2d = npz_data[f"{b_key}_bedrock_ac"] if has_ac else bedrock_2d

        for i in range(n_lat):
            bedrock_grid.append([round(float(bedrock_2d[i, j]), 1) for j in range(n_lon)])
            bedrock_eng_grid.append([round(float(bedrock_eng_2d[i, j]), 1) for j in range(n_lon)])
            bedrock_ac_grid.append([round(float(bedrock_ac_2d[i, j]), 1) for j in range(n_lon)])

        basins_voxels[b_key] = {
            "bounds": {
                "lon_min": float(lons.min()), "lon_max": float(lons.max()),
                "lat_min": float(lats.min()), "lat_max": float(lats.max()),
                "depth_min": float(depths.min()), "depth_max": float(depths.max())
            },
            "lats": [round(float(x), 4) for x in lats],
            "lons": [round(float(x), 4) for x in lons],
            "depths": [round(float(x), 1) for x in depths],
            "points": pts,
            "bedrock_grid": bedrock_grid,
            "bedrock_eng_grid": bedrock_eng_grid,
            "bedrock_ac_grid": bedrock_ac_grid
        }
        print(f"  [{basins_meta[b_key]['name']}]: {len(pts):,} 個高密度體素點")

    payload = {
        "boreholes": boreholes,
        "faults": faults,
        "basins_meta": basins_meta,
        "basins_voxels": basins_voxels
    }

    print(f"\n輸出數據至 {out_js}...")
    json_str = json.dumps(payload, ensure_ascii=False)
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("window.TAIWAN_PORTAL_DATA = " + json_str + ";\n")

    size_mb = out_js.stat().st_size / 1024 / 1024
    print(f"3D 地圖門戶資料封包已儲存至: {out_js} (體積: {size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
