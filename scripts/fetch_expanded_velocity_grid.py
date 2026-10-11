"""從中研院 TEC TWtomo API 下載西南臺灣 (嘉南平原與雲嘉南地區) 3D 震波速度網格。

範圍擴展：
  緯度：22.95°N (臺南市南界/二仁溪) ~ 23.85°N (雲林北界/濁水溪)
  經度：120.00°E (西部海域/臺灣海峽) ~ 120.75°E (麓山帶前緣/阿里山山麓)
  深度：0 ~ 5 km (涵蓋前第四紀基盤、北港基底高區與臺南深沉積盆地)

輸出：
  data/interim/chianan_velocity_3d.csv
"""
import argparse
import io
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
    time.sleep(3)
    
    z_resp = session.get(zip_url, headers=headers, verify=False)
    z_resp.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(z_resp.content)) as z:
        with z.open("vel.out") as f:
            df = pd.read_csv(f, sep=r'\s+', header=None)
            
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

def build_expanded_grid(out_dir: str = "data/interim", num_slices: int = 10):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "chianan_velocity_3d.csv"
    
    # 涵蓋雲林、嘉義、臺南全域
    lats = np.linspace(22.95, 23.85, num_slices)
    lon_min, lon_max = 120.00, 120.75
    
    all_dfs = []
    print(f"開始抓取西南臺灣 (嘉南平原) 3D 速度網格 ({num_slices} 條剖面切片: 22.95°N ~ 23.85°N)...")
    for i, lat in enumerate(lats):
        print(f"[{i+1}/{num_slices}] 正在請求緯度 {lat:.3f}°N ({lon_min}°E -> {lon_max}°E)...")
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
    print(f"\n全域西南臺灣網格建置完成！總點數: {len(full_df)}")
    print(f"存檔路徑: {csv_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=10, help="剖面條數")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()
    build_expanded_grid(out_dir=args.out, num_slices=args.slices)
