"""測試與下載中研院 TEC 震波速度模型 (Taiwan Velocity Model, TVM)。

測試端點：https://tecdc.earth.sinica.edu.tw/TWtomo/php/runProfvelorPlot.php
"""
import io
import os
import sys
import time
import urllib.parse
import urllib3
import zipfile
import pandas as pd
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_tec_api(lat1=23.3, lon1=120.1, lat2=23.6, lon2=120.6, depth=5, model="LEE2025"):
    api_url = "https://tecdc.earth.sinica.edu.tw/TWtomo/php/runProfvelorPlot.php"
    base_url = "https://tecdc.earth.sinica.edu.tw/TWtomo/"
    secret_str = f"{lat1},{lon1},{lat2},{lon2}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
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
    print(f"發送請求至 TEC API: {secret_str}, model={model}, depth={depth}km...")
    response = session.post(api_url, data=payload, headers=headers, timeout=60, verify=False)
    print("API 回應碼:", response.status_code)
    try:
        res = response.json()
        print("API 回應 JSON:", res)
    except Exception as e:
        print("非 JSON 回應:", response.text[:300])
        return False

    dir_id = str(res.get("dir", res.get("1", "")))
    if not dir_id:
        print("未獲取目錄 ID")
        return False
        
    zip_path = f"output/{dir_id}/slice.zip"
    full_zip_url = urllib.parse.urljoin(base_url, zip_path)
    print(f"等待 3 秒後下載 slice.zip: {full_zip_url}")
    time.sleep(3)
    
    zip_resp = session.get(full_zip_url, headers=headers, verify=False)
    print("ZIP 下載狀態:", zip_resp.status_code, "長度:", len(zip_resp.content))
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
        print("ZIP 內檔案:", z.namelist())
        for fname in z.namelist():
            if fname.endswith(('.txt', '.out', '.dat', '.csv')):
                with z.open(fname) as f:
                    df = pd.read_csv(f, sep=r'\s+', header=None)
                    print(f"檔案 {fname} 行數: {len(df)}, 欄數: {df.shape[1]}")
                    print("前 3 筆:\n", df.head(3))
                    return True
    return False

if __name__ == "__main__":
    test_tec_api()
