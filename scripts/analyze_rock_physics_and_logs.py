"""井下地球物理電測 (Gamma, SP, N16, N64) 與岩石物理模板 (RPT) 及瓦爾特相律轉移矩陣分析。

本腳本：
1. 讀取全臺 384 口深水觀測井之真實地球物理電測數據 (data/raw/hydro/phys/*.json)。
2. 將百萬筆真實井下電測 (自然伽瑪 GAMMA、自然電位 SP、16吋與64吋法向電阻率 N16/N64) 對齊地層岩性紀錄。
3. 結合 3D 震波速度場 (Vp, Vs, Vp/Vs, Poisson's ratio)，繪製國際科展級「岩石物理模板 (Rock Physics Template, RPT)」：
   - Gamma Ray vs. Log(N64 電阻率) 跨尺度散佈與岩性分離度分析。
   - 聲波阻抗 (Ip) vs. Vp/Vs 比值之岩石物理鑑別板。
   - 沉積相地球物理物性箱形分佈圖。
4. 統計全臺深井沉積相垂直演化轉移矩陣，驗證「瓦爾特相律 (Walther's Law)」沉積旋迴定量性。
5. 產出報告圖件：
   - reports/rock_physics_template.png
   - reports/walther_transition_matrix.png
   - reports/phys_logging_benchmark_table.csv
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

L2_NAMES = {
    2: "黏土/泥 (Clay/Mud)",
    3: "粉土/粉砂 (Silt)",
    4: "細砂 (Fine Sand)",
    5: "粗砂 (Coarse Sand)",
    6: "砂泥互層 (Sand-Mud Mixed)",
    7: "礫石 (Gravel)"
}

L2_COLORS = {
    "黏土/泥 (Clay/Mud)": "#8B4513",
    "粉土/粉砂 (Silt)": "#CD853F",
    "細砂 (Fine Sand)": "#E6B800",
    "粗砂 (Coarse Sand)": "#D2691E",
    "砂泥互層 (Sand-Mud Mixed)": "#4682B4",
    "礫石 (Gravel)": "#2E8B57"
}

def extract_and_align_well_logs(
    phys_dir: Path,
    strata_path: Path,
    sample_stride: int = 2
) -> pd.DataFrame:
    """快速對齊真實井測數據與沉積地層紀錄。"""
    print(f"載入地層分層記錄 {strata_path}...")
    strata = pd.read_csv(strata_path)
    # 僅篩選六大主要沉積相 (L2: 2 ~ 7)
    valid_strata = strata[strata["l2"].isin(list(L2_NAMES.keys()))].copy()
    strata_by_station = dict(tuple(valid_strata.groupby("station")))

    phys_files = list(phys_dir.glob("*.json"))
    print(f"讀取 {len(phys_files)} 口深水觀測井真實電測 JSON...")

    records = []
    for f in phys_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                j = json.load(fp)
            d = j.get("data", {})
            st = d.get("STATION_NMBR")
            if not st or st not in strata_by_station:
                continue
            
            st_df = strata_by_station[st].sort_values("top_m").reset_index(drop=True)
            tops = st_df["top_m"].values
            bots = st_df["bottom_m"].values
            l2_ids = st_df["l2"].values

            data_rows = d.get("Data", [])
            for row in data_rows[::sample_stride]:
                depth = row.get("d")
                gr = row.get("GAMMA")
                r64 = row.get("N64")
                r16 = row.get("N16")
                sp = row.get("SP")

                if depth is None or (gr is None and r64 is None):
                    continue

                # 快速二分搜尋匹配區間
                idx = np.searchsorted(bots, depth, side="right")
                if idx < len(tops) and tops[idx] <= depth < bots[idx]:
                    records.append({
                        "station": st,
                        "depth_m": depth,
                        "gamma": gr,
                        "n64": r64,
                        "n16": r16,
                        "sp": sp,
                        "lithology_id": l2_ids[idx]
                    })
        except Exception:
            continue

    df_logs = pd.DataFrame(records)
    print(f"井測對齊完成！總共成功匹配 {len(df_logs):,} 筆真實深井電測實測點。")
    return df_logs


def compute_walther_transition_matrix(strata_path: Path) -> pd.DataFrame:
    """依據全臺真實深井分層計算垂直沉積相轉移機率矩陣 (Walther's Law)。"""
    strata = pd.read_csv(strata_path)
    valid = strata[strata["l2"].isin(list(L2_NAMES.keys()))].copy()

    classes = [2, 3, 4, 5, 6, 7]
    class_map = {c: i for i, c in enumerate(classes)}
    n_c = len(classes)
    counts = np.zeros((n_c, n_c), dtype=float)

    for _, well_df in valid.groupby("station"):
        well_sorted = well_df.sort_values("top_m")
        l2_seq = well_sorted["l2"].values
        for i in range(len(l2_seq) - 1):
            src = l2_seq[i]
            dst = l2_seq[i + 1]
            if src != dst:
                counts[class_map[src], class_map[dst]] += 1.0

    # 列正規化 (轉移機率)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_prob = counts / row_sums

    labels = [L2_NAMES[c] for c in classes]
    return pd.DataFrame(trans_prob, index=labels, columns=labels)


def plot_rock_physics_and_logs(
    df_logs: pd.DataFrame,
    dataset_path: Path,
    out_rpt_path: Path,
    out_walther_path: Path,
    out_table_path: Path
):
    """繪製國際科展發表級岩石物理模板 (RPT) 與瓦爾特相律轉移圖。"""
    # 清洗井測異常值 (物理邊界過濾)
    clean = df_logs[
        (df_logs["gamma"].notnull()) & (df_logs["gamma"] > 5) & (df_logs["gamma"] < 250) &
        (df_logs["n64"].notnull()) & (df_logs["n64"] > 1) & (df_logs["n64"] < 600)
    ].copy()
    clean["lithology_name"] = clean["lithology_id"].map(L2_NAMES)

    # 1. 輸出物理物性基準統計表
    summary = clean.groupby("lithology_name")[["gamma", "n64"]].agg([
        ("count", "count"),
        ("median", "median"),
        ("q25", lambda x: np.percentile(x, 25)),
        ("q75", lambda x: np.percentile(x, 75))
    ]).round(1)
    summary.to_csv(out_table_path, encoding="utf-8-sig")
    print(f"地球物理井測基準表已輸出至: {out_table_path}")

    # 2. 繪製四格岩石物理與井測模板 (Rock Physics Template)
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # (a) Gamma Ray vs. Log(N64) 散佈圖與群集分離 (Crossplot)
    ax_a = axes[0, 0]
    # 取樣以防繪圖過密
    sample_vis = clean.groupby("lithology_name").sample(n=min(len(clean)//6, 1200), random_state=42, replace=False)
    for name, grp in sample_vis.groupby("lithology_name"):
        c = L2_COLORS.get(name, "#888888")
        ax_a.scatter(grp["gamma"], grp["n64"], label=name, color=c, alpha=0.55, edgecolors="none", s=22)

    ax_a.set_yscale("log")
    ax_a.set_ylim(5, 500)
    ax_a.set_xlim(15, 180)
    ax_a.set_xlabel("自然伽瑪值 Gamma Ray (API / CPS)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("長法向視電阻率 N64 (Ω·m, Log Scale)", fontsize=11, fontweight="bold")
    ax_a.set_title("(a) 全臺深井真實電測自然伽瑪 vs. 電阻率交繪圖 (Well-Log RPT)", fontsize=12, fontweight="bold")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)
    # 加入物理解釋標註
    ax_a.annotate("泥岩/黏土相\n(高 Gamma、低電阻率)", xy=(120, 18), xytext=(125, 35),
                  arrowprops=dict(arrowstyle="->", color="#8B4513", lw=1.5), fontsize=10, fontweight="bold", color="#8B4513")
    ax_a.annotate("粗砂與礫石相\n(低 Gamma、高電阻率)", xy=(45, 140), xytext=(22, 220),
                  arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.5), fontsize=10, fontweight="bold", color="#2E8B57")

    # (b) 震波阻抗 Ip vs. Vp/Vs 比值模板 (Seismic Rock Physics Template)
    ax_b = axes[0, 1]
    if dataset_path.exists():
        df_ds = pd.read_csv(dataset_path)
        valid_ds = df_ds[df_ds["lithology_id"].isin(list(L2_NAMES.keys()))].copy()
        valid_ds["lithology_name"] = valid_ds["lithology_id"].map(L2_NAMES)
        # 估計密度 (Gardner 經驗公式): rho = 1.74 * (Vp * 1e-3)^0.25 (g/cm^3)
        vp_kms = valid_ds["vp_ms"] / 1000.0
        rho = 1.74 * (vp_kms ** 0.25)
        # 聲波阻抗 Ip = rho * Vp (km/s * g/cm^3)
        valid_ds["ip"] = rho * vp_kms
        
        sample_ds = valid_ds.groupby("lithology_name").sample(n=min(len(valid_ds)//6, 1200), random_state=42, replace=False)
        for name, grp in sample_ds.groupby("lithology_name"):
            c = L2_COLORS.get(name, "#888888")
            ax_b.scatter(grp["ip"], grp["vp_vs_ratio"], label=name, color=c, alpha=0.55, edgecolors="none", s=22)

        ax_b.set_xlabel("聲波阻抗 Acoustic Impedance Ip (km/s · g/cm³)", fontsize=11, fontweight="bold")
        ax_b.set_ylabel("縱橫波速比 Vp/Vs Ratio", fontsize=11, fontweight="bold")
        ax_b.set_title("(b) 3D 震波速度場與岩性聲阻抗關係 (Seismic Rock Physics Template)", fontsize=12, fontweight="bold")
        ax_b.grid(True, linestyle="--", alpha=0.4)
        ax_b.legend(loc="upper right", frameon=True, fontsize=9)

    # (c) Gamma Ray 各沉積相箱形圖 (Boxplot)
    ax_c = axes[1, 0]
    order = list(L2_COLORS.keys())
    data_gr = [clean[clean["lithology_name"] == name]["gamma"].dropna().values for name in order]
    bp_c = ax_c.boxplot(data_gr, patch_artist=True, showfliers=False, widths=0.5)
    for patch, name in zip(bp_c['boxes'], order):
        patch.set_facecolor(L2_COLORS[name])
        patch.set_alpha(0.7)
    for median in bp_c['medians']:
        median.set(color='black', linewidth=1.5)
    ax_c.set_xticks(range(1, len(order) + 1))
    ax_c.set_xticklabels(order, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax_c.set_ylabel("自然伽瑪 Gamma Ray (API)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("")
    ax_c.set_title("(c) 六大沉積相真實自然伽瑪響應統計分佈", fontsize=12, fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.4)

    # (d) Resistivity N64 各沉積相箱形圖 (Boxplot)
    ax_d = axes[1, 1]
    data_n64 = [clean[clean["lithology_name"] == name]["n64"].dropna().values for name in order]
    bp_d = ax_d.boxplot(data_n64, patch_artist=True, showfliers=False, widths=0.5)
    for patch, name in zip(bp_d['boxes'], order):
        patch.set_facecolor(L2_COLORS[name])
        patch.set_alpha(0.7)
    for median in bp_d['medians']:
        median.set(color='black', linewidth=1.5)
    ax_d.set_yscale("log")
    ax_d.set_ylim(8, 400)
    ax_d.set_xticks(range(1, len(order) + 1))
    ax_d.set_xticklabels(order, rotation=20, ha="right", fontsize=9, fontweight="bold")
    ax_d.set_ylabel("64吋法向電阻率 N64 (Ω·m, Log Scale)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("")
    ax_d.set_title("(d) 六大沉積相真實井下電阻率響應統計分佈", fontsize=12, fontweight="bold")
    ax_d.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    plt.suptitle("臺灣沉積盆地深井地球物理電測實測約束與岩石物理模板 (Rock Physics & Wireline Logs Template)", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(out_rpt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"岩石物理模板圖件已成功儲存至: {out_rpt_path}")

    # 3. 繪製瓦爾特相律轉移矩陣熱力圖 (Walther's Law Facies Transition Matrix)
    strata_path = dataset_path.parents[0] / ".." / "interim" / "taiwan_strata.csv"
    if not strata_path.exists():
        strata_path = Path("data/interim/taiwan_strata.csv")
    trans_df = compute_walther_transition_matrix(strata_path)

    fig_w, ax_w = plt.subplots(figsize=(10, 8))
    mat = trans_df.values * 100.0
    im = ax_w.imshow(mat, cmap="YlGnBu", aspect="auto")
    cbar = fig_w.colorbar(im, ax=ax_w)
    cbar.set_label("垂直沈積相轉移機率 Transition Probability (%)", fontsize=11, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val > 28 else "black"
            ax_w.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax_w.set_xticks(range(len(trans_df.columns)))
    ax_w.set_yticks(range(len(trans_df.index)))
    ax_w.set_xticklabels(trans_df.columns, rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax_w.set_yticklabels(trans_df.index, fontsize=10, fontweight="bold")
    ax_w.set_title("臺灣主要沉積盆地垂直相序轉移矩陣（瓦爾特相律定量化檢定）\nWalther's Law Stratigraphic Transition Matrix", fontsize=13, fontweight="bold", pad=15)
    ax_w.set_xlabel("上覆地層沈積相 (Overlying Lithology)", fontsize=11, fontweight="bold")
    ax_w.set_ylabel("下伏地層沈積相 (Underlying Lithology)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig_w.savefig(out_walther_path, dpi=300, bbox_inches="tight")
    plt.close(fig_w)
    print(f"瓦爾特相律轉移矩陣圖件已成功儲存至: {out_walther_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    phys_dir = root / "data" / "raw" / "hydro" / "phys"
    strata_path = root / "data" / "interim" / "taiwan_strata.csv"
    dataset_path = root / "data" / "processed" / "dataset_v1.csv"
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("啟動地球物理電測岩石物理模板 (RPT) 與瓦爾特相律分析")
    print("=================================================================")
    
    df_logs = extract_and_align_well_logs(phys_dir, strata_path, sample_stride=2)
    
    out_rpt = rep_dir / "rock_physics_template.png"
    out_walther = rep_dir / "walther_transition_matrix.png"
    out_table = rep_dir / "phys_logging_benchmark_table.csv"

    plot_rock_physics_and_logs(
        df_logs=df_logs,
        dataset_path=dataset_path,
        out_rpt_path=out_rpt,
        out_walther_path=out_walther,
        out_table_path=out_table
    )

if __name__ == "__main__":
    main()
