"""臺灣全島水文地質觀測井資料庫建置模組 (Taiwan-Wide Hydro Borehole Database Builder)。

自 data/raw/hydro/HydroStation.json (GeoJSON) 與 data/raw/hydro/rockrecord/ 萃取：
1. 全台 423 口深層水文地質觀測井之基本資料 (經緯度、TWD97、高程、深度、行政區、地質分區)。
2. 全台已下載之深井地層分層資料 (依據 L1/L2 標準岩性映射)。

輸出：
- data/interim/taiwan_boreholes.csv (全台 423 站完整座標與地質屬性)
- data/interim/taiwan_strata.csv (已解析深井地層層段)
"""
import json
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.labels.hydro_codes import hydro_to_l1, hydro_to_l2
from tages.labels.lithology import L1_NAMES, L2_NAMES

def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "hydro"
    interim_dir = root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    station_json = raw_dir / "HydroStation.json"
    if not station_json.exists():
        print(f"找不到 {station_json}！")
        return

    st_data = json.loads(station_json.read_text(encoding="utf-8"))
    features = st_data["data"]["features"]
    print(f"讀取全台觀測井 GeoJSON: 共 {len(features)} 站")

    # 1. 建立全台觀測井基本資料表 (taiwan_boreholes.csv)
    have_rock_files = {p.stem: p for p in (raw_dir / "rockrecord").glob("*.json")}
    bh_rows = []
    strata_rows = []

    for f in features:
        p = f["properties"]
        sid = p["STATION_NMBR"]
        coords = f["geometry"]["coordinates"]
        lon, lat = coords[0], coords[1]
        
        has_file = sid in have_rock_files
        well_d = float(p.get("Well_Depth") or 0.0)
        elev = float(p.get("CUR_ELEVATION") or 0.0)

        bh_rows.append({
            "station": sid,
            "name": p.get("STATION_NAME", ""),
            "city": p.get("City_Name", ""),
            "wsect": p.get("Wsect_Name", ""),
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "x97": round(float(p.get("x97") or 0.0), 2),
            "y97": round(float(p.get("y97") or 0.0), 2),
            "elev_m": round(elev, 2),
            "well_depth_m": round(well_d, 2),
            "has_phys": bool(p.get("WellPhys")),
            "has_c14": bool(p.get("WellCarbonAge")),
            "year": int(p.get("EXEYEAR") or 0),
            "has_rock_record": bool(p.get("HydroRockRecord")),
            "is_downloaded": has_file
        })

        # 2. 若已有地層 JSON，解析岩性分層
        if has_file:
            try:
                rr = json.loads(have_rock_files[sid].read_text(encoding="utf-8"))
                records = rr.get("data", {}).get("Data", []) or []
                for r in records:
                    code = str(r.get("TYPE_OF_ROCK", "")).strip()
                    top = float(r.get("TOP_DEPTH") or 0.0)
                    bot = float(r.get("BOTTOM_DEPTH") or 0.0)
                    if bot <= top:
                        continue
                    l1 = hydro_to_l1(code)
                    l2 = hydro_to_l2(code)
                    strata_rows.append({
                        "station": sid,
                        "top_m": round(top, 2),
                        "bottom_m": round(bot, 2),
                        "rock_code": code,
                        "l1": l1,
                        "l1_name": L1_NAMES.get(l1, "Unknown"),
                        "l2": l2,
                        "l2_name": L2_NAMES.get(l2, "Unknown")
                    })
            except Exception as e:
                pass

    df_bh = pd.DataFrame(bh_rows)
    df_st = pd.DataFrame(strata_rows)

    out_bh = interim_dir / "taiwan_boreholes.csv"
    out_st = interim_dir / "taiwan_strata.csv"

    df_bh.to_csv(out_bh, index=False, encoding="utf-8-sig")
    df_st.to_csv(out_st, index=False, encoding="utf-8-sig")

    print(f"\n臺灣全島水文觀測井資料庫建置完成:")
    print(f"  總井數: {len(df_bh)} 口 (存檔: {out_bh})")
    print(f"  已解析地層紀錄: {len(df_st):,} 筆 (存檔: {out_st})")
    print(f"\n各縣市觀測井分佈統計:")
    print(df_bh["city"].value_counts().to_string())

if __name__ == "__main__":
    main()
