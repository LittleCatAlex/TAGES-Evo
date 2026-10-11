"""將水文地質鑽井原始 JSON 轉為統一 schema。

輸出：
  data/interim/hydro_boreholes.csv  井位（TWD97、WGS84、高程、井深、水區）
  data/interim/hydro_strata.csv     分層（top_m, bottom_m, rock_code, l1, l1_name）
  data/interim/hydro_phys_summary.csv 井測可用深度範圍與欄位覆蓋

用法：python scripts/build_hydro_strata.py --raw data/raw/hydro --out data/interim
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.labels.hydro_codes import hydro_to_l1, hydro_to_l2  # noqa: E402
from tages.labels.lithology import L1_NAMES, L2_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/hydro")
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = json.loads((raw / "HydroStation.json").read_text(encoding="utf-8"))
    have = {p.stem for p in (raw / "rockrecord").glob("*.json")}

    bh_rows, strata_rows, phys_rows = [], [], []
    for f in st["data"]["features"]:
        p = f["properties"]
        sid = p["STATION_NMBR"]
        if sid not in have:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        bh_rows.append({
            "station": sid, "name": p["STATION_NAME"], "city": p["City_Name"],
            "wsect": p["Wsect_Name"], "lon": lon, "lat": lat,
            "x97": p["x97"], "y97": p["y97"], "elev_m": p["CUR_ELEVATION"],
            "well_depth_m": p["Well_Depth"], "has_phys": p["WellPhys"],
            "has_c14": p["WellCarbonAge"], "year": p["EXEYEAR"],
        })
        rr = json.loads((raw / "rockrecord" / f"{sid}.json").read_text(encoding="utf-8"))
        for r in rr["data"]["Data"] or []:
            code = r["TYPE_OF_ROCK"]
            l1 = hydro_to_l1(code)
            l2 = hydro_to_l2(code)
            strata_rows.append({"station": sid, "top_m": r["TOP_DEPTH"], "bottom_m": r["BOTTOM_DEPTH"],
                                "rock_code": code, "l1": l1, "l1_name": L1_NAMES[l1],
                                "l2": l2, "l2_name": L2_NAMES[l2]})
        pf = raw / "phys" / f"{sid}.json"
        if pf.exists():
            pj = json.loads(pf.read_text(encoding="utf-8"))["data"]["Data"] or []
            if pj:
                d = pd.DataFrame(pj)
                phys_rows.append({"station": sid, "d_min": d.d.min(), "d_max": d.d.max(), "n": len(d),
                                  **{f"{c}_cover": float(d[c].notna().mean()) for c in ("SP", "N16", "N64", "GAMMA")}})

    bh = pd.DataFrame(bh_rows)
    sr = pd.DataFrame(strata_rows)
    bh.to_csv(out / "hydro_boreholes.csv", index=False, encoding="utf-8-sig")
    sr.to_csv(out / "hydro_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phys_rows).to_csv(out / "hydro_phys_summary.csv", index=False, encoding="utf-8-sig")

    print("井數:", len(bh), "分層數:", len(sr))
    print(bh.groupby("city").size().to_string())
    sr["thick"] = (sr.bottom_m - sr.top_m).clip(lower=0)
    print("\n各 L1 累計厚度 (m) 與佔比:")
    g = sr.groupby("l1_name").thick.sum().sort_values(ascending=False)
    print(pd.concat([g.round(1), (g / g.sum() * 100).round(1).rename("pct")], axis=1).to_string())
    print("\n井深分布:", bh.well_depth_m.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
