"""下載臺灣水文地質資訊系統 (hydro.geologycloud.tw) 平原區鑽井資料。

端點（由網站前端 JS 逆向確認，POST）：
  /data/P/HydroStation                 全部平原區鑽井點位 (GeoJSON)
  /data/P/rockrecord?station=<STATION> 岩性紀錄
  /data/P/phys?station=<STATION>       井測紀錄
  /data/P/carbon_age?station=<STATION> C14 定年

用法：
  python scripts/fetch_hydro.py --bbox 120.0 23.2 120.7 23.75 --out data/raw/hydro
支援斷點續跑（已存在的檔案會略過）。
"""
import argparse
import json
import time
from pathlib import Path

import requests

BASE = "https://hydro.geologycloud.tw"


def post(session, path, retries=3):
    for i in range(retries):
        try:
            r = session.post(BASE + path, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON1", "LAT1", "LON2", "LAT2"))
    ap.add_argument("--out", default="data/raw/hydro")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("rockrecord", "phys", "carbon_age"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    st = post(s, "/data/P/HydroStation")
    (out / "HydroStation.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    feats = st["data"]["features"]
    print(f"全部站數: {len(feats)}")

    if args.bbox:
        x1, y1, x2, y2 = args.bbox
        feats = [f for f in feats
                 if x1 <= f["geometry"]["coordinates"][0] <= x2
                 and y1 <= f["geometry"]["coordinates"][1] <= y2]
    print(f"範圍內站數: {len(feats)}")

    for f in feats:
        p = f["properties"]
        sid = p["STATION_NMBR"]
        for sub, flag in (("rockrecord", p.get("HydroRockRecord")),
                          ("phys", p.get("WellPhys")),
                          ("carbon_age", p.get("WellCarbonAge"))):
            if not flag:
                continue
            fp = out / sub / f"{sid}.json"
            if fp.exists():
                continue
            try:
                data = post(s, f"/data/P/{sub}?station={sid}")
                fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print(f"[失敗] {sub} {sid}: {e}")
            time.sleep(args.delay)
        print("ok", sid, p.get("STATION_NAME"), p.get("Well_Depth"))


if __name__ == "__main__":
    main()
