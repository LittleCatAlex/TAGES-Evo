"""從中研院 TEC TWtomo API 下載全臺灣 (北部、中部、南部、東部與恆春) 3D 震波速度網格。

整合既有嘉南網格與全台代表性地質剖面切片：
1. 臺北盆地與基隆 (25.10°N, 121.20°E ~ 121.90°E)
2. 桃園/新竹與蘭陽平原 (24.80°N, 120.90°E ~ 121.90°E)
3. 苗栗與臺中盆地 (24.40°N, 120.50°E ~ 121.50°E)
4. 彰化與南投埔里 (24.10°N, 120.30°E ~ 121.50°E)
5. 西南臺灣嘉南平原 (既有 10 條密集切片: 22.95°N ~ 23.85°N, 120.00°E ~ 120.75°E)
6. 高雄與屏東平原 (22.60°N, 120.20°E ~ 120.80°E)
7. 恆春半島 (22.20°N, 120.60°E ~ 121.00°E)
8. 花東縱谷板塊縫合帶 (23.50°N, 121.10°E ~ 121.60°E)

輸出：
  data/interim/taiwan_velocity_3d.csv
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from fetch_expanded_velocity_grid import fetch_slice

SECTIONS = [
    {"name": "Taipei Basin & Keelung", "lat1": 25.10, "lon1": 121.20, "lat2": 25.10, "lon2": 121.90},
    {"name": "Taoyuan/Hsinchu to Yilan", "lat1": 24.80, "lon1": 120.90, "lat2": 24.80, "lon2": 121.90},
    {"name": "Miaoli to Taichung Basin", "lat1": 24.40, "lon1": 120.50, "lat2": 24.40, "lon2": 121.50},
    {"name": "Changhua to Central Range", "lat1": 24.10, "lon1": 120.30, "lat2": 24.10, "lon2": 121.50},
    {"name": "Kaohsiung to Pingtung Plain", "lat1": 22.60, "lon1": 120.20, "lat2": 22.60, "lon2": 120.80},
    {"name": "Hengchun Peninsula", "lat1": 22.20, "lon1": 120.60, "lat2": 22.20, "lon2": 121.00},
    {"name": "Hualien-Taitung Longitudinal Valley", "lat1": 23.50, "lon1": 121.10, "lat2": 23.50, "lon2": 121.60}
]

def main():
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "data" / "interim" / "taiwan_velocity_3d.csv"
    chianan_csv = root / "data" / "interim" / "chianan_velocity_3d.csv"

    all_dfs = []
    # 1. 載入既有西南臺灣切片 (1,520 點)
    if chianan_csv.exists():
        print(f"載入既有西南臺灣切片: {chianan_csv}")
        df_cn = pd.read_csv(chianan_csv)
        all_dfs.append(df_cn)

    # 2. 依序抓取全台其餘主要構造分區
    print("\n開始抓取全台灣主要地體構造切片 (TEC LEE2025 模型)...")
    for sec in SECTIONS:
        print(f"正在請求 [{sec['name']}]: ({sec['lat1']}, {sec['lon1']}) -> ({sec['lat2']}, {sec['lon2']})...")
        try:
            df = fetch_slice(sec["lat1"], sec["lon1"], sec["lat2"], sec["lon2"], depth=5, model="LEE2025")
            all_dfs.append(df)
            print(f"  成功獲取 {len(df)} 點速度資料！")
        except Exception as e:
            print(f"  請求失敗 ({sec['name']}): {e}")
        time.sleep(2)

    full_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["lat", "lon", "depth_m"])
    full_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n全台灣 3D 震波速度網格建置完成！總點數: {len(full_df):,} 點")
    print(f"存檔路徑: {out_csv}")

if __name__ == "__main__":
    main()
