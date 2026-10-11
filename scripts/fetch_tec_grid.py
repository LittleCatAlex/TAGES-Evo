"""從中研院 TEC TWtomo API 下載嘉義 3D 震波速度網格 (Taiwan Velocity Model)。

原理：
對嘉義地區 (Lat: 23.20 ~ 23.75, Lon: 120.0 ~ 120.7) 進行多條等緯度剖面抽樣，
結合深度 (0 ~ 5 km)，建構全區 3D 速度網格 (Vp, Vs)，並建立三維線性內插器。
"""
import argparse
import io
import os
import sys
import time
import urllib.parse
import urllib3
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = "https://tecdc.earth.sinica.edu.tw/TWtomo/php/runProfvelorPlot.php"
BASE_URL = "https://tecdc.earth.sinica.edu.tw/TWtomo/"

def fetch_slice(lat1: float, lon1: float, lat2: float, lon2: float, depth: int = 5, model: str = "LEE2025") -> pd.DataFrame:
    """獲取單條剖面的速度數據。"""
    secret_str = f"{lat1:.3f},{lon1:.3f},{lat2:.3f},{lon2:.3f}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://tecdc.earth.sinica.edu.tw/TWtomo/",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    payload = {
        "secret": secret_str,
        "eq_width": "5",
        "model": model,
        "depth": str(depth),
        "grid": "2",
        "cpt": "0",
        "seisoption": "1",
        "cptopt": "seis",
        "cptinvert": "0"
    }
    
    session = requests.Session()
    resp = session.post(API_URL, data=payload, headers=headers, timeout=60, verify=False)
    resp.raise_for_status()
    res = resp.json()
    
    dir_id = str(res.get("dir", res.get("1", "")))
    if not dir_id:
        raise ValueError(f"無法取得 dir_id: {res}")
        
    zip_path = f"output/{dir_id}/slice.zip"
    zip_url = urllib.parse.urljoin(BASE_URL, zip_path)
    time.sleep(3)  # 等待伺服器打包
    
    z_resp = session.get(zip_url, headers=headers, verify=False)
    z_resp.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(z_resp.content)) as z:
        with z.open("vel.out") as f:
            df = pd.read_csv(f, sep=r'\s+', header=None)
            
    # 指定欄位: Col 1: Depth (km), Col 2: Vp (km/s), Col 4: Vs (km/s), Col 8: Lon, Col 9: Lat
    cleaned = pd.DataFrame({
        "lat": df[9],
        "lon": df[8],
        "depth_km": df[1],
        "depth_m": df[1] * 1000.0,
        "vp_kms": df[2],
        "vs_kms": df[4],
        "vp_ms": df[2] * 1000.0,
        "vs_ms": df[4] * 1000.0
    })
    cleaned["vp_vs_ratio"] = cleaned["vp_ms"] / np.maximum(cleaned["vs_ms"], 1e-3)
    return cleaned

def build_chiayi_grid(out_dir: str = "data/interim", num_slices: int = 8):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "chiayi_velocity_3d.csv"
    
    lats = np.linspace(23.20, 23.75, num_slices)
    lon_min, lon_max = 120.00, 120.70
    
    all_dfs = []
    print(f"開始抓取嘉義 3D 速度網格 ({num_slices} 條剖面切片)...")
    for i, lat in enumerate(lats):
        print(f"[{i+1}/{num_slices}] 正在請求緯度 {lat:.3f}N ({lon_min}E -> {lon_max}E)...")
        try:
            df = fetch_slice(lat, lon_min, lat, lon_max, depth=5, model="LEE2025")
            all_dfs.append(df)
            print(f"  成功獲取 {len(df)} 點速度資料 (深度範圍: {df.depth_m.min()}~{df.depth_m.max()}m)")
        except Exception as e:
            print(f"  請求失敗: {e}")
        time.sleep(2)
        
    if not all_dfs:
        print("未獲取任何剖面資料！")
        return
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"\n全區網格建置完成！總點數: {len(full_df)}")
    print(f"存檔路徑: {csv_file}")
    print(full_df.describe().round(2).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=8, help="剖面條數")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()
    build_chiayi_grid(out_dir=args.out, num_slices=args.slices)
