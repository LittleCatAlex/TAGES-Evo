"""建立全臺 384 口水文觀測井之真實地球物理電測（Gamma, SP, N16, N64）對齊資料集。

本腳本執行 Phase A 之步驟 A2：
1. 讀取全臺水文觀測井真實電測 JSON (data/raw/hydro/phys/*.json)。
2. 結合地層分層記錄 (data/interim/taiwan_strata.csv) 與井位資訊 (data/interim/taiwan_boreholes.csv)，
   將深度取樣點精準匹配至沉積相岩性 (L2 十大岩性標籤)。
3. 執行井內自然伽瑪射線分位數標準化 (Per-well Shale Volume Index, I_GR)：
   I_GR = (GR - GR_5%) / (GR_95% - GR_5%)
   解決全臺跨包商儀器計數尺度 (API vs. CPS) 混雜導致的粉土異常，確立單調沉積物性序列。
4. 計算第三系/第四系鬆散沉積物 Larionov 非線性泥質體積分率 V_sh：
   V_sh = 0.083 * (2^(3.7 * I_GR) - 1.0)
5. 計算視電阻率深度比值 R_ratio = N64 / N16 (反映泥漿濾液侵入剖面與地層孔隙滲透性)。
6. 匯出對齊後的物理資料集：
   - data/processed/petrophysics_aligned_dataset.csv
   - data/processed/petrophysics_aligned_dataset.parquet (若支援)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# L2 岩性對應字典 (沉積相六大類與基盤)
L2_NAMES = {
    2: "黏土/泥 (Clay/Mud)",
    3: "粉土/粉砂 (Silt)",
    4: "細砂 (Fine Sand)",
    5: "粗砂 (Coarse Sand)",
    6: "砂泥互層 (Sand-Mud Mixed)",
    7: "礫石 (Gravel)",
    8: "泥岩 (Mudstone)",
    9: "砂岩 (Sandstone)",
    10: "結晶基盤 (Basement)"
}


def build_petrophysics_dataset(
    phys_dir: str | Path = "data/raw/hydro/phys",
    strata_path: str | Path = "data/interim/taiwan_strata.csv",
    boreholes_path: str | Path = "data/interim/taiwan_boreholes.csv",
    out_csv_path: str | Path = "data/processed/petrophysics_aligned_dataset.csv",
    sample_stride: int = 1
) -> pd.DataFrame:
    """提取、標準化並對齊全臺井下地球物理電測資料。"""
    phys_p = Path(phys_dir)
    strata_p = Path(strata_path)
    bh_p = Path(boreholes_path)
    out_csv = Path(out_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== [Phase A2] 構建地球物理電測對齊資料集 ===")
    print(f"1. 讀取地層分層記錄: {strata_p}")
    strata_df = pd.read_csv(strata_p)
    strata_by_station = dict(tuple(strata_df.groupby("station")))

    bh_dict = {}
    if bh_p.exists():
        print(f"2. 讀取井位座標與海拔高程: {bh_p}")
        bh_df = pd.read_csv(bh_p)
        for _, r in bh_df.iterrows():
            bh_dict[str(r["station"])] = {
                "station_name": r.get("name", ""),
                "city": r.get("city", ""),
                "wsect": r.get("wsect", ""),
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "x_twd97": float(r.get("x97", r.get("x_twd97", 0.0))),
                "y_twd97": float(r.get("y97", r.get("y_twd97", 0.0))),
                "elev_m": float(r.get("elev_m", 0.0))
            }

    phys_files = sorted(list(phys_p.glob("*.json")))
    print(f"3. 掃描深水觀測井電測 JSON，共發現 {len(phys_files)} 口井...")

    all_well_records = []
    matched_wells_count = 0

    for f in phys_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                j = json.load(fp)
            d = j.get("data", {})
            st = str(d.get("STATION_NMBR", "")).strip()
            if not st or st not in strata_by_station:
                continue

            st_df = strata_by_station[st].sort_values("top_m").reset_index(drop=True)
            tops = st_df["top_m"].values
            bots = st_df["bottom_m"].values
            l1_ids = st_df["l1"].values
            l2_ids = st_df["l2"].values
            l1_names = st_df["l1_name"].values
            l2_names = st_df["l2_name"].values

            bh_info = bh_dict.get(st, {
                "station_name": "", "city": "", "wsect": "",
                "lon": 0.0, "lat": 0.0, "x_twd97": 0.0, "y_twd97": 0.0, "elev_m": 0.0
            })

            data_rows = d.get("Data", [])
            if not data_rows:
                continue

            well_pts = []
            for row in data_rows[::sample_stride]:
                depth = row.get("d")
                gr = row.get("GAMMA")
                r64 = row.get("N64")
                r16 = row.get("N16")
                sp = row.get("SP")

                if depth is None:
                    continue
                # 至少要有 Gamma 或 電阻率
                if gr is None and r64 is None and r16 is None:
                    continue

                # 快速二分搜尋匹配岩性層段
                idx = np.searchsorted(bots, depth, side="right")
                if idx < len(tops) and tops[idx] <= depth <= bots[idx]:
                    well_pts.append({
                        "station": st,
                        "station_name": bh_info["station_name"],
                        "city": bh_info["city"],
                        "wsect": bh_info["wsect"],
                        "lon": bh_info["lon"],
                        "lat": bh_info["lat"],
                        "x_twd97": bh_info["x_twd97"],
                        "y_twd97": bh_info["y_twd97"],
                        "elev_m": bh_info["elev_m"],
                        "depth_m": round(float(depth), 2),
                        "z_msl": round(bh_info["elev_m"] - float(depth), 2),
                        "gamma": float(gr) if gr is not None else np.nan,
                        "n64": float(r64) if r64 is not None else np.nan,
                        "n16": float(r16) if r16 is not None else np.nan,
                        "sp": float(sp) if sp is not None else np.nan,
                        "lithology_l1_id": int(l1_ids[idx]),
                        "lithology_l1_name": str(l1_names[idx]),
                        "lithology_l2_id": int(l2_ids[idx]),
                        "lithology_l2_name": str(l2_names[idx])
                    })

            if not well_pts:
                continue

            df_single = pd.DataFrame(well_pts)

            # --- 4. 井內分位數標準化自然伽瑪 (Per-Well Normalization) ---
            gr_valid = df_single["gamma"].dropna()
            # 排除明顯儀器零值或極端雜訊
            gr_clean = gr_valid[(gr_valid > 1.0) & (gr_valid < 350.0)]
            if len(gr_clean) >= 20:
                p5 = float(np.percentile(gr_clean, 5))
                p95 = float(np.percentile(gr_clean, 95))
                delta = max(5.0, p95 - p5)
                # 泥質含量指數 I_GR (截斷至 [0.0, 1.0])
                df_single["i_gr"] = np.clip((df_single["gamma"] - p5) / delta, 0.0, 1.0).round(4)
                # Larionov (1969) 未固結第三系/第四系沉積物泥質體積分率 V_sh
                # V_sh = 0.083 * (2^(3.7 * I_GR) - 1.0)
                v_sh = 0.083 * (np.power(2.0, 3.7 * df_single["i_gr"]) - 1.0)
                df_single["v_shale"] = np.clip(v_sh, 0.0, 1.0).round(4)
            else:
                df_single["i_gr"] = np.nan
                df_single["v_shale"] = np.nan

            # --- 5. 視電阻率比值 R_ratio = N64 / N16 ---
            valid_res = (df_single["n64"] > 0.1) & (df_single["n16"] > 0.1)
            df_single["r_ratio_64_16"] = np.where(
                valid_res,
                np.round(df_single["n64"] / np.maximum(df_single["n16"], 0.1), 3),
                np.nan
            )

            all_well_records.append(df_single)
            matched_wells_count += 1

        except Exception as e:
            continue

    if not all_well_records:
        raise RuntimeError("未能成功匹配任何水文觀測井之地球物理電測資料！")

    df_dataset = pd.concat(all_well_records, ignore_index=True)
    print(f"\n電測對齊完畢！")
    print(f"- 成功匹配觀測井數: {matched_wells_count} 口井")
    print(f"- 總物理電測採樣點: {len(df_dataset):,} 筆")

    # 匯出至 CSV
    df_dataset.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"對齊資料集已成功儲存至: {out_csv}")

    # 嘗試匯出至 Parquet
    parquet_p = out_csv.with_suffix(".parquet")
    try:
        df_dataset.to_parquet(parquet_p, index=False)
        print(f"Parquet 格式已同步儲存至: {parquet_p}")
    except Exception as e:
        print(f"Parquet 儲存略過: {e}")

    # 輸出六大沉積相統計摘要
    print("\n--- 全臺主要沉積相之地球物理電測基準統計 (中位數與四分位距) ---")
    valid_sub = df_dataset[df_dataset["lithology_l2_id"].isin(list(L2_NAMES.keys()))].copy()
    valid_sub["lithology_name"] = valid_sub["lithology_l2_id"].map(L2_NAMES)
    
    summary = valid_sub.groupby("lithology_name").agg(
        pts=("depth_m", "count"),
        gr_median=("gamma", "median"),
        i_gr_median=("i_gr", "median"),
        v_shale_median=("v_shale", "median"),
        n64_median=("n64", "median"),
        n16_median=("n16", "median"),
        r_ratio_median=("r_ratio_64_16", "median")
    ).round(3)
    print(summary.to_string())

    return df_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="構建地球物理電測對齊資料集")
    parser.add_argument("--phys_dir", default="data/raw/hydro/phys", help="電測 JSON 目錄")
    parser.add_argument("--strata", default="data/interim/taiwan_strata.csv", help="地層紀錄 CSV")
    parser.add_argument("--boreholes", default="data/interim/taiwan_boreholes.csv", help="井位 CSV")
    parser.add_argument("--out", default="data/processed/petrophysics_aligned_dataset.csv", help="輸出 CSV")
    parser.add_argument("--stride", type=int, default=1, help="深度採樣步長 (預設 1 即全量保留)")
    args = parser.parse_args()

    build_petrophysics_dataset(
        phys_dir=args.phys_dir,
        strata_path=args.strata,
        boreholes_path=args.boreholes,
        out_csv_path=args.out,
        sample_stride=args.stride
    )
