"""?Ｙ??儔 MVP ??雿???(manifest)??
- 霈??GeoScrapter ??all_taiwan_geotech_holes.csv
- 蝭拚?儔撟喳??喲?撅勗葆 bbox
- ?? H 瑽?chart_photos / core_photos 瑼?嚗?閮?摮?行??梁???撗拙??抒?
- 靘?瘛梁策鈭?OCR ?芸?蝝?瘛勗??芸?嚗?
瑼??澆?嚗?  chart: <proj_no>_<hole_no>_Chart_<n>.png
  core : <proj_no>_<hole_no>_<top>m-<bot>m.jpg

?冽?嚗?  python scripts/build_manifest.py --csv ../repos/GeoScrapter/all_taiwan_geotech_holes.csv \
      --photos "H:/?啁?蝘?" --out data/interim/chiayi_manifest.csv
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BBOX = (120.0, 23.2, 120.7, 23.75)  # lon1, lat1, lon2, lat2


def scan_names(folder: Path, pattern: re.Pattern):
    found = defaultdict(list)
    if not folder.exists():
        return found
    with os.scandir(folder) as it:
        for e in it:
            m = pattern.match(e.name)
            if m:
                found[(m.group("proj"), m.group("hole"))].append(e.name)
    return found


def priority(depth):
    if depth >= 100:
        return 1
    if depth >= 60:
        return 2
    if depth >= 30:
        return 3
    return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", default="data/interim/chiayi_manifest.csv")
    ap.add_argument("--bbox", nargs=4, type=float, default=BBOX)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"proj_no": str, "hole_no": str}, low_memory=False)
    x1, y1, x2, y2 = args.bbox
    df = df[df.coord_x.between(x1, x2) & df.coord_y.between(y1, y2)].copy()
    df["total_depth"] = pd.to_numeric(df["total_depth"], errors="coerce")
    print("bbox ?批???", len(df))

    photos = Path(args.photos)
    chart_re = re.compile(r"^(?P<proj>.+?)_(?P<hole>.+)_Chart_(?P<n>\d+)\.png$", re.I)
    core_re = re.compile(r"^(?P<proj>[^_]+)_(?P<hole>.+)_[\d.]+m-[\d.]+m\.jpg$", re.I)
    charts = scan_names(photos / "chart_photos", chart_re)
    cores = scan_names(photos / "core_photos", core_re)
    print("?梁??????典):", len(charts), "撗拙?摮(?典):", len(cores))

    keys = list(zip(df.proj_no, df.hole_no))
    df["n_chart"] = [len(charts.get(k, [])) for k in keys]
    df["n_core"] = [len(cores.get(k, [])) for k in keys]
    df["chart_file"] = [charts[k][0] if k in charts else "" for k in keys]
    df["has_chart"] = df.n_chart > 0
    df["has_core"] = df.n_core > 0
    df["ocr_priority"] = df.total_depth.fillna(0).map(priority)
    df = df.sort_values(["ocr_priority", "total_depth"], ascending=[True, False])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("頛詨:", args.out)
    print("????", int(df.has_chart.sum()), "/", len(df))
    print(df.groupby("ocr_priority").agg(holes=("hole_no", "size"),
                                         with_chart=("has_chart", "sum"),
                                         with_core=("has_core", "sum")))


if __name__ == "__main__":
    main()

