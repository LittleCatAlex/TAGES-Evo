"""臺灣公開工程地質鑽孔資料庫解析與特徵資料集建置模組 (Engineering Boreholes Pipeline).

從原始 OCR JSON (data/borehole_final_results.json) 重新建置：
1. data/interim/school_boreholes.csv
2. data/interim/school_strata.csv
3. data/interim/taiwan_boreholes_combined.csv
4. data/interim/taiwan_strata_combined.csv
5. data/processed/dataset_school.csv
6. data/processed/dataset_combined.csv
7. reports/ocr_error_audit.json
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tages.geo.crs import convert, in_taiwan_wgs84
from tages.labels.lithology import classify_l1, classify_l2, L1_NAMES, L2_NAMES
from tages.features.interpolator import VelocityModel3D

WESTERN_DEFORMATION_FRONT = np.array([
    [121.50, 25.05], [121.25, 24.85], [120.95, 24.45], [120.72, 24.15],
    [120.58, 23.65], [120.48, 23.35], [120.42, 22.85], [120.45, 22.40], [120.75, 21.95]
])

WESTERN_COASTLINE = np.array([
    [121.50, 25.25], [121.15, 25.10], [120.95, 24.75], [120.62, 24.35],
    [120.30, 23.95], [120.12, 23.55], [120.08, 23.10], [120.20, 22.50], [120.65, 21.90]
])


def point_to_polyline_dist_km(lons: np.ndarray, lats: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    lat_mean = np.radians(np.mean(lats))
    scale_y = 111.0
    scale_x = 111.0 * np.cos(lat_mean)
    px = lons * scale_x
    py = lats * scale_y
    poly_x = polyline[:, 0] * scale_x
    poly_y = polyline[:, 1] * scale_y
    min_dists = np.full(len(lons), np.inf)
    for i in range(len(polyline) - 1):
        x1, y1 = poly_x[i], poly_y[i]
        x2, y2 = poly_x[i + 1], poly_y[i + 1]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0:
            d = np.hypot(px - x1, py - y1)
        else:
            t = np.clip(((px - x1) * dx + (py - y1) * dy) / l2, 0.0, 1.0)
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = np.hypot(px - proj_x, py - proj_y)
        min_dists = np.minimum(min_dists, d)
    return np.round(min_dists, 2)


def parse_school_boreholes(json_path: Path):
    """解析工程地質鑽孔 OCR 結果 JSON"""
    print(f"--> 載入工程地質鑽孔 OCR 資料庫: {json_path}")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"    讀取 {len(data):,} 張柱狀圖，耗時 {time.time() - t0:.2f} 秒")

    raw_wells = {}
    for filename, info in data.items():
        if "_Chart_" in filename:
            w_id = filename.split("_Chart_")[0]
        elif "_chart_" in filename.lower():
            w_id = filename.lower().split("_chart_")[0]
        else:
            w_id = filename

        binfo = info.get("borehole_info", {})
        raw_bid = binfo.get("borehole_id", "UNKNOWN").strip()
        if w_id not in raw_wells:
            raw_wells[w_id] = {
                "raw_bid": raw_bid,
                "location": binfo.get("location", ""),
                "datum": str(binfo.get("coordinates", {}).get("datum", "TWD97")).strip().lower(),
                "N": binfo.get("coordinates", {}).get("N"),
                "E": binfo.get("coordinates", {}).get("E"),
                "elevation_m": binfo.get("elevation_m"),
                "date": binfo.get("date", ""),
                "strata": []
            }
        for s in info.get("strata", []):
            raw_wells[w_id]["strata"].append((s.get("depth_m"), str(s.get("description", ""))))

    bh_rows = []
    st_rows = []

    for w_id, w in sorted(raw_wells.items()):
        station_id = f"SCH_{w_id}"
        try:
            e = float(w["E"])
            n = float(w["N"])
            datum = w["datum"]
            if 118 <= e <= 123 and 21 <= n <= 26:
                lon, lat = e, n
                x97, y97 = convert(lon, lat, "wgs84", "twd97")
            elif 118 <= n <= 123 and 21 <= e <= 26:
                lon, lat = n, e
                x97, y97 = convert(lon, lat, "wgs84", "twd97")
            else:
                if e > 1000000:
                    e /= 10.0
                if n > 10000000:
                    n /= 10.0
                src_crs = "twd67" if ("67" in datum) else "twd97"
                lon, lat = convert(e, n, src_crs, "wgs84")
                x97, y97 = convert(e, n, src_crs, "twd97")
        except Exception:
            continue

        if not (120.0 <= lon <= 122.5 and 21.8 <= lat <= 25.5):
            continue

        elev = w["elevation_m"]
        try:
            elev_f = float(elev)
            if -50 <= elev_f <= 3952:
                elev = elev_f
            else:
                elev = 15.0
        except Exception:
            elev = 15.0

        seen_depths = set()
        sorted_s = []
        for d, desc in sorted(w["strata"], key=lambda x: x[0] if x[0] is not None else 0):
            if d is not None and 0 < d <= 350.0 and d not in seen_depths:
                seen_depths.add(d)
                sorted_s.append((d, desc))

        if not sorted_s:
            continue

        prev_d = 0.0
        max_depth = 0.0
        for d, desc in sorted_s:
            top_m = round(prev_d, 2)
            bot_m = round(d, 2)
            thick = bot_m - top_m
            if thick <= 0 or thick > 80.0:
                prev_d = d
                continue
            prev_d = d
            max_depth = max(max_depth, bot_m)
            l1 = classify_l1(desc)
            l2 = classify_l2(desc)

            st_rows.append({
                "station": station_id,
                "top_m": top_m,
                "bottom_m": bot_m,
                "rock_code": desc[:60],
                "l1": l1,
                "l1_name": L1_NAMES.get(l1, "Unknown"),
                "l2": l2,
                "l2_name": L2_NAMES.get(l2, "Unknown")
            })

        year = 1990
        d_str = str(w.get("date", ""))
        if "103" in d_str or "2014" in d_str:
            year = 2014
        elif "104" in d_str or "2015" in d_str:
            year = 2015
        elif "098" in d_str or "2009" in d_str:
            year = 2009

        loc_desc = str(w.get("location", "")).replace("\n", " ").strip()
        bh_rows.append({
            "station": station_id,
            "name": f"學校孔_{w_id}_{loc_desc}"[:80],
            "city": "學校工程孔",
            "wsect": "工程地質",
            "lon": round(lon, 6),
            "lat": round(lat, 6),
            "x97": round(x97, 2),
            "y97": round(y97, 2),
            "elev_m": round(elev, 2),
            "well_depth_m": round(max_depth, 2),
            "has_phys": False,
            "has_c14": False,
            "year": year,
            "has_rock_record": True,
            "is_downloaded": True
        })

    df_bh = pd.DataFrame(bh_rows)
    df_st = pd.DataFrame(st_rows)
    print(f"--> 解析完成: {len(df_bh):,} 口有效鑽井，{len(df_st):,} 層地層分層")
    return df_bh, df_st, len(data)


def build_sampled_features(df_bh: pd.DataFrame, df_st: pd.DataFrame, velocity_grid_path: Path, sample_step_m: float = 1.0):
    """依 1.0m 取樣步長建置 14 維特徵資料集"""
    print("--> 執行 1.0m 等距取樣與特徵工程...")
    t0 = time.time()
    bh_dict = df_bh.set_index("station").to_dict("index")

    samples = []
    # 使用 itertuples 極速迭代
    for row in df_st.itertuples(index=False):
        st_id = row.station
        if st_id not in bh_dict:
            continue
        bh_info = bh_dict[st_id]
        top, bot = row.top_m, row.bottom_m
        thick = max(0.1, bot - top)
        if row.l2 == 0 or thick > 80.0 or bot > 350.0:
            continue

        if thick <= sample_step_m:
            depths = [(top + bot) / 2.0]
        else:
            depths = list(np.arange(top + 0.5, bot, sample_step_m))
            if not depths:
                depths = [(top + bot) / 2.0]

        lon_val = bh_info["lon"]
        lat_val = bh_info["lat"]
        x97_val = bh_info["x97"]
        y97_val = bh_info["y97"]
        elev_val = bh_info["elev_m"]
        st_name = bh_info["name"]
        city_val = bh_info["city"]
        wsect_val = bh_info["wsect"]

        for d in depths:
            samples.append((
                st_id, st_name, city_val, wsect_val,
                lon_val, lat_val, x97_val, y97_val, elev_val,
                round(d, 2), round(elev_val - d, 2),
                top, bot, round(thick, 2),
                row.rock_code, row.l1, row.l1_name, row.l2, row.l2_name
            ))

    col_names = [
        "station_id", "station_name", "city", "wsect",
        "lon", "lat", "x_twd97", "y_twd97", "elevation_m",
        "depth_m", "z_msl", "layer_top_m", "layer_bottom_m", "layer_thickness_m",
        "rock_code", "lithology_l1_id", "lithology_l1_name", "lithology_id", "lithology_name"
    ]
    df_samples = pd.DataFrame(samples, columns=col_names)
    print(f"    細分取樣完成: 共 {len(df_samples):,} 筆樣本，耗時 {time.time() - t0:.2f} 秒")

    print("--> 查詢中研院 TEC LEE2025 3D 震波速度場與斷層/海岸幾何距離...")
    t1 = time.time()
    vm = VelocityModel3D(str(velocity_grid_path))
    vel_feats = vm.query(
        lat=df_samples["lat"].values,
        lon=df_samples["lon"].values,
        depth_m=df_samples["depth_m"].values
    )

    df_samples["vp_ms"] = np.round(vel_feats["vp"], 1)
    df_samples["vs_ms"] = np.round(vel_feats["vs"], 1)
    df_samples["vp_vs_ratio"] = np.round(vel_feats["vp_vs_ratio"], 3)
    df_samples["dvs_dz"] = np.round(vel_feats["dvs_dz"], 4)

    r2 = df_samples["vp_vs_ratio"] ** 2
    df_samples["poisson_ratio"] = np.round(0.5 * (r2 - 2) / np.maximum(r2 - 1, 1e-4), 3)
    df_samples["dist_to_fault_km"] = point_to_polyline_dist_km(
        df_samples["lon"].values, df_samples["lat"].values, WESTERN_DEFORMATION_FRONT
    )
    df_samples["dist_to_coast_km"] = point_to_polyline_dist_km(
        df_samples["lon"].values, df_samples["lat"].values, WESTERN_COASTLINE
    )
    compaction_vs = 160.0 * (1.0 + df_samples["depth_m"] / 8.0) ** 0.32
    df_samples["compaction_vs_ms"] = np.round(compaction_vs, 1)
    df_samples["vs_residual_ms"] = np.round(df_samples["vs_ms"] - compaction_vs, 1)
    print(f"    速度內插與特徵計算完成，耗時 {time.time() - t1:.2f} 秒")

    return df_samples


def audit_ocr_errors(error_json_path: Path, valid_charts_count: int, reports_dir: Path):
    """分析 error.json 中的錯誤分類與品質控制指標"""
    if not error_json_path.exists():
        return None
    print(f"--> 分析 OCR 品質控制異常紀錄: {error_json_path}")
    with open(error_json_path, "r", encoding="utf-8") as f:
        err_data = json.load(f)

    reasons = {}
    total_err = len(err_data)
    for _, v in err_data.items():
        if not isinstance(v, dict):
            continue
        r = v.get("error_reason", "unknown")
        r_key = r.split(":")[0] if ":" in r else r
        r_key = r_key[:50].strip()
        reasons[r_key] = reasons.get(r_key, 0) + 1

    sorted_reasons = sorted(reasons.items(), key=lambda x: -x[1])
    total_scanned = int(valid_charts_count + total_err)
    audit_summary = {
        "total_scanned_charts": total_scanned,
        "valid_retained_charts": int(valid_charts_count),
        "total_rejected_charts": int(total_err),
        "primary_rejection_reasons": {r: count for r, count in sorted_reasons[:15]},
        "quality_control_policy": f"自原始掃描之 {total_scanned:,} 張圖檔中，嚴格排除 {total_err:,} 張非地層柱狀圖或深度異常紀錄，最終保留 {valid_charts_count:,} 張有效工程地質鑽探柱狀圖進入模型訓練"
    }

    out_audit_path = reports_dir / "ocr_error_audit.json"
    with open(out_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    print(f"   已產出 OCR 品質控制稽核報告: {out_audit_path}")
    return audit_summary


def main():
    interim_dir = ROOT / "data" / "interim"
    processed_dir = ROOT / "data" / "processed"
    reports_dir = ROOT / "reports"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = ROOT / "data" / "borehole_final_results.json"
    error_json_path = ROOT / "data" / "error.json"
    velocity_grid = interim_dir / "taiwan_velocity_3d.csv"

    # 1. 解析原始鑽孔 JSON
    df_school_bh, df_school_st, valid_charts = parse_school_boreholes(json_path)

    # 儲存 school_boreholes.csv & school_strata.csv
    df_school_bh.to_csv(interim_dir / "school_boreholes.csv", index=False, encoding="utf-8-sig")
    df_school_st.to_csv(interim_dir / "school_strata.csv", index=False, encoding="utf-8-sig")
    print(f"   已儲存: {interim_dir / 'school_boreholes.csv'} ({len(df_school_bh):,} 口)")
    print(f"   已儲存: {interim_dir / 'school_strata.csv'} ({len(df_school_st):,} 層)")

    # 2. 合併全臺觀測井
    orig_bh_path = interim_dir / "taiwan_boreholes.csv"
    orig_st_path = interim_dir / "taiwan_strata.csv"
    df_orig_bh = pd.read_csv(orig_bh_path)
    df_orig_st = pd.read_csv(orig_st_path)

    df_comb_bh = pd.concat([df_orig_bh, df_school_bh], ignore_index=True).drop_duplicates(subset=["station"])
    df_comb_st = pd.concat([df_orig_st, df_school_st], ignore_index=True)
    df_comb_bh.to_csv(interim_dir / "taiwan_boreholes_combined.csv", index=False, encoding="utf-8-sig")
    df_comb_st.to_csv(interim_dir / "taiwan_strata_combined.csv", index=False, encoding="utf-8-sig")
    print(f"   已儲存: taiwan_boreholes_combined.csv (總井數: {len(df_comb_bh):,} 口)")
    print(f"   已儲存: taiwan_strata_combined.csv (總層數: {len(df_comb_st):,} 層)")

    # 3. 建立 1.0m 特徵資料集
    df_school_samples = build_sampled_features(df_school_bh, df_school_st, velocity_grid, sample_step_m=1.0)
    df_school_samples.to_csv(processed_dir / "dataset_school.csv", index=False, encoding="utf-8-sig")
    print(f"   已儲存: dataset_school.csv ({len(df_school_samples):,} 筆樣本)")

    # 4. 合併水文基準 dataset_v1.csv
    orig_dataset_path = processed_dir / "dataset_v1.csv"
    df_orig_dataset = pd.read_csv(orig_dataset_path)
    df_comb_dataset = pd.concat([df_orig_dataset, df_school_samples], ignore_index=True)
    df_comb_dataset.to_csv(processed_dir / "dataset_combined.csv", index=False, encoding="utf-8-sig")
    print(f"   已儲存: dataset_combined.csv (總樣本: {len(df_comb_dataset):,} 筆)")

    # 5. 產出 OCR 品質控制稽核報告
    audit_ocr_errors(error_json_path, valid_charts, reports_dir)

    print("\n=================================================================")
    print(" 新工程地質鑽孔資料庫建置與特徵資料集更新完成！")
    print("=================================================================")


if __name__ == "__main__":
    main()
