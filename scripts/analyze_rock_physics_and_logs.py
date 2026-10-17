"""井下地球物理電測 (Gamma, SP, N16, N64) 與岩石物理模板 (RPT) 及瓦爾特相律轉移矩陣分析。

本腳本執行 Phase A 之成果產出（圖 A-1 與基準統計表）：
1. 讀取全臺 384 口水文觀測井之真實地球物理電測對齊資料集 (data/processed/petrophysics_aligned_dataset.csv)。
2. 計算全臺主要沉積相之地球物理電測基準統計 (中位數、Q25、Q75、平均值)，輸出至：
   - reports/phys_logging_benchmark_table.csv
3. 繪製國際科展發表級「岩石物理模板 (Rock Physics Template, RPT)」四格圖：
   - (a) 井內標準化自然伽瑪泥質含量指數 I_GR vs. Log(N64 電阻率) 物理交繪圖 (Well-Log RPT)。
   - (b) 聲波阻抗 Ip vs. Vp/Vs 比值之地震岩石物理鑑別板 (Seismic RPT)。
   - (c) 六大沉積相真實泥質含量指數 I_GR 箱形分佈圖 (展現粒徑分選物理階梯)。
   - (d) 六大沉積相真實井下電阻率 N64 箱形分佈圖。
   輸出至: reports/rock_physics_template.png
4. 統計全臺深井沉積相垂直演化轉移矩陣，驗證「瓦爾特相律 (Walther's Law)」沉積旋迴定量性：
   輸出至: reports/walther_transition_matrix.png
"""
from __future__ import annotations

import argparse
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

ORDER_FACIES = [
    "粗砂 (Coarse Sand)",
    "礫石 (Gravel)",
    "細砂 (Fine Sand)",
    "砂泥互層 (Sand-Mud Mixed)",
    "粉土/粉砂 (Silt)",
    "黏土/泥 (Clay/Mud)"
]


def generate_phys_benchmark_table(df_logs: pd.DataFrame, out_path: Path):
    """計算並輸出沉積相之地球物理電測基準統計表。"""
    valid = df_logs[df_logs["lithology_l2_id"].isin(list(L2_NAMES.keys()))].copy()
    valid["lithology_name"] = valid["lithology_l2_id"].map(L2_NAMES)

    metrics = []
    for name, grp in valid.groupby("lithology_name"):
        row = {
            "沉積岩性 (Facies)": name,
            "樣本數 (Points)": len(grp),
            "I_GR 泥質指數 (中位數)": grp["i_gr"].median(),
            "I_GR (Q25)": grp["i_gr"].quantile(0.25),
            "I_GR (Q75)": grp["i_gr"].quantile(0.75),
            "V_shale 泥質體積分率 (中位數)": grp["v_shale"].median(),
            "N64 電阻率 Ω·m (中位數)": grp["n64"].median(),
            "N64 (Q25)": grp["n64"].quantile(0.25),
            "N64 (Q75)": grp["n64"].quantile(0.75),
            "N16 電阻率 Ω·m (中位數)": grp["n16"].median(),
            "N64/N16 視電阻率比 (中位數)": grp["r_ratio_64_16"].median(),
            "原始 Gamma API (中位數)": grp["gamma"].median(),
        }
        metrics.append(row)

    df_tab = pd.DataFrame(metrics)
    # 按沉積相粗細排序
    sorter = {name: i for i, name in enumerate(ORDER_FACIES)}
    df_tab["sort_idx"] = df_tab["沉積岩性 (Facies)"].map(sorter)
    df_tab = df_tab.sort_values("sort_idx").drop(columns=["sort_idx"]).round(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_tab.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"地球物理井測基準統計表已輸出至: {out_path}")
    print(df_tab.to_string())


def plot_rock_physics_template(
    df_logs: pd.DataFrame,
    dataset_path: Path,
    out_rpt_path: Path
):
    """繪製國際科展發表級四格岩石物理模板 (Rock Physics Template)。"""
    # 篩選有效且非離群的電測資料 (物理邊界過濾)
    clean = df_logs[
        (df_logs["lithology_l2_id"].isin(list(L2_NAMES.keys()))) &
        (df_logs["i_gr"].notnull()) & (df_logs["i_gr"] >= 0.0) & (df_logs["i_gr"] <= 1.0) &
        (df_logs["n64"].notnull()) & (df_logs["n64"] > 1.0) & (df_logs["n64"] < 600.0)
    ].copy()
    clean["lithology_name"] = clean["lithology_l2_id"].map(L2_NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # -------------------------------------------------------------
    # (a) 自然伽瑪泥質含量指數 I_GR vs. Log(N64) 電阻率交繪圖 (Well-Log RPT)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    # 取樣以保證可讀性與繪圖效能
    sample_vis = clean.groupby("lithology_name").sample(
        n=min(len(clean) // 6, 1500), random_state=42, replace=False
    )
    for name, grp in sample_vis.groupby("lithology_name"):
        c = L2_COLORS.get(name, "#888888")
        ax_a.scatter(
            grp["i_gr"], grp["n64"], label=name, color=c, alpha=0.50, edgecolors="none", s=24
        )

    ax_a.set_yscale("log")
    ax_a.set_ylim(5, 500)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel("井內標準化泥質含量指數 Shale Volume Index $I_{\\text{GR}}$", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("長法向視電阻率 N64 ($\\Omega\\cdot\\text{m}$, Log Scale)", fontsize=11, fontweight="bold")
    ax_a.set_title("(a) 全臺深井電測泥質含量 vs. 視電阻率交繪圖 (Well-Log RPT)", fontsize=12, fontweight="bold")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)

    # 地學物理解釋標註
    ax_a.annotate(
        "泥岩/黏土相\n(高泥質含量 $I_{\\text{GR}} > 0.65$、低電阻率)",
        xy=(0.75, 20), xytext=(0.55, 10),
        arrowprops=dict(arrowstyle="->", color="#8B4513", lw=1.8),
        fontsize=10, fontweight="bold", color="#8B4513"
    )
    ax_a.annotate(
        "粗砂與礫石相\n(低泥質含量 $I_{\\text{GR}} < 0.45$、高電阻率)",
        xy=(0.28, 180), xytext=(0.05, 240),
        arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.8),
        fontsize=10, fontweight="bold", color="#2E8B57"
    )

    # -------------------------------------------------------------
    # (b) 震波阻抗 Ip vs. Vp/Vs 比值模板 (Seismic Rock Physics Template)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    if dataset_path.exists():
        df_ds = pd.read_csv(dataset_path)
        valid_ds = df_ds[df_ds["lithology_id"].isin(list(L2_NAMES.keys()))].copy()
        valid_ds["lithology_name"] = valid_ds["lithology_id"].map(L2_NAMES)
        # Gardner (1974) 估計沉積物密度: rho = 1.74 * (Vp * 1e-3)^0.25 (g/cm^3)
        vp_kms = valid_ds["vp_ms"] / 1000.0
        rho = 1.74 * (vp_kms ** 0.25)
        valid_ds["ip"] = rho * vp_kms  # 聲波阻抗 (km/s * g/cm^3)

        sample_ds = valid_ds.groupby("lithology_name").sample(
            n=min(len(valid_ds) // 6, 1200), random_state=42, replace=False
        )
        for name, grp in sample_ds.groupby("lithology_name"):
            c = L2_COLORS.get(name, "#888888")
            ax_b.scatter(grp["ip"], grp["vp_vs_ratio"], label=name, color=c, alpha=0.55, edgecolors="none", s=22)

        ax_b.set_xlabel("聲波阻抗 Acoustic Impedance $I_p$ ($\\text{km/s} \\cdot \\text{g/cm}^3$)", fontsize=11, fontweight="bold")
        ax_b.set_ylabel("縱橫波速比 $V_p/V_s$ Ratio", fontsize=11, fontweight="bold")
        ax_b.set_title("(b) 3D 震波速度場與岩性聲阻抗關係 (Seismic Rock Physics Template)", fontsize=12, fontweight="bold")
        ax_b.grid(True, linestyle="--", alpha=0.4)
        ax_b.legend(loc="upper right", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # (c) 六大沉積相真實泥質含量指數 I_GR 箱形圖 (Boxplot)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    data_igr = [clean[clean["lithology_name"] == name]["i_gr"].dropna().values for name in ORDER_FACIES]
    bp_c = ax_c.boxplot(data_igr, patch_artist=True, showfliers=False, widths=0.52)
    for patch, name in zip(bp_c['boxes'], ORDER_FACIES):
        patch.set_facecolor(L2_COLORS[name])
        patch.set_alpha(0.75)
    for median in bp_c['medians']:
        median.set(color='black', linewidth=1.8)
    
    # 繪製中位數演變趨勢折線
    medians = [np.median(vals) for vals in data_igr]
    ax_c.plot(range(1, len(ORDER_FACIES) + 1), medians, color="red", linestyle="--", marker="o", lw=1.5, label="中位數演進趨勢 (粒徑分選規律)")

    ax_c.set_xticks(range(1, len(ORDER_FACIES) + 1))
    ax_c.set_xticklabels(ORDER_FACIES, rotation=18, ha="right", fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel("井內泥質含量指數 $I_{\\text{GR}}$ (無因次)", fontsize=11, fontweight="bold")
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title("(c) 六大沉積相真實泥質含量指數階梯分佈 (驗證粒徑沉積規律)", fontsize=12, fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.4)
    ax_c.legend(loc="upper left", fontsize=9)

    # -------------------------------------------------------------
    # (d) 六大沉積相真實井下電阻率 N64 箱形圖 (Boxplot)
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    data_n64 = [clean[clean["lithology_name"] == name]["n64"].dropna().values for name in ORDER_FACIES]
    bp_d = ax_d.boxplot(data_n64, patch_artist=True, showfliers=False, widths=0.52)
    for patch, name in zip(bp_d['boxes'], ORDER_FACIES):
        patch.set_facecolor(L2_COLORS[name])
        patch.set_alpha(0.75)
    for median in bp_d['medians']:
        median.set(color='black', linewidth=1.8)

    ax_d.set_yscale("log")
    ax_d.set_ylim(8, 450)
    ax_d.set_xticks(range(1, len(ORDER_FACIES) + 1))
    ax_d.set_xticklabels(ORDER_FACIES, rotation=18, ha="right", fontsize=9.5, fontweight="bold")
    ax_d.set_ylabel("64吋法向視電阻率 N64 ($\\Omega\\cdot\\text{m}$, Log Scale)", fontsize=11, fontweight="bold")
    ax_d.set_title("(d) 六大沉積相真實井下電阻率響應分佈 (粗粒高阻 vs. 細粒低阻)", fontsize=12, fontweight="bold")
    ax_d.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    plt.suptitle("臺灣主要沉積盆地井下地球物理電測約束與岩石物理模板 (Rock Physics & Wireline Logs Template)", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_rpt_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_rpt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"岩石物理模板圖件已成功儲存至: {out_rpt_path}")


def compute_and_plot_walther_matrix(strata_path: Path, out_path: Path):
    """計算並繪製全臺深井垂直地層轉移機率熱力圖 (Walther's Law)。"""
    strata = pd.read_csv(strata_path)
    valid = strata[strata["l2"].isin(list(L2_NAMES.keys()))].copy()

    classes = [5, 7, 4, 6, 3, 2]  # 粗砂, 礫石, 細砂, 互層, 粉土, 黏土
    class_map = {c: i for i, c in enumerate(classes)}
    n_c = len(classes)
    counts = np.zeros((n_c, n_c), dtype=float)

    for _, well_df in valid.groupby("station"):
        well_sorted = well_df.sort_values("top_m")
        l2_seq = well_sorted["l2"].values
        for i in range(len(l2_seq) - 1):
            src = l2_seq[i]
            dst = l2_seq[i + 1]
            if src in class_map and dst in class_map and src != dst:
                counts[class_map[src], class_map[dst]] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_prob = counts / row_sums

    labels = [L2_NAMES[c].split()[0] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trans_prob, cmap="YlGnBu", vmin=0.0, vmax=0.55)

    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_c))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("下覆沉積相 (Deeper Facies / Transition Target)", fontsize=11, fontweight="bold")
    ax.set_ylabel("上覆沉積相 (Shallower Facies / Transition Source)", fontsize=11, fontweight="bold")
    ax.set_title("全臺沉積盆地實測垂直相序轉移機率矩陣\n(Walther's Law Sedimentary Facies Transition)", fontsize=12, fontweight="bold", pad=12)

    for i in range(n_c):
        for j in range(n_c):
            val = trans_prob[i, j]
            color = "white" if val > 0.28 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("轉移機率 Transition Probability", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"瓦爾特相律轉移矩陣圖已成功儲存至: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="分析井下地球物理電測與岩石物理模板")
    parser.add_argument("--logs_csv", default="data/processed/petrophysics_aligned_dataset.csv", help="電測對齊 CSV")
    parser.add_argument("--dataset_csv", default="data/processed/dataset_v1.csv", help="3D 速度場特徵資料集")
    parser.add_argument("--strata_csv", default="data/interim/taiwan_strata.csv", help="地層紀錄 CSV")
    parser.add_argument("--out_rpt", default="reports/rock_physics_template.png", help="RPT 圖片路徑")
    parser.add_argument("--out_walther", default="reports/walther_transition_matrix.png", help="瓦爾特轉移圖路徑")
    parser.add_argument("--out_table", default="reports/phys_logging_benchmark_table.csv", help="基準統計表路徑")
    args = parser.parse_args()

    logs_p = Path(args.logs_csv)
    if not logs_p.exists():
        print(f"找不到電測對齊資料集 {logs_p}，請先執行 scripts/build_petrophysics_dataset.py！")
        return

    print(f"載入電測對齊資料集: {logs_p}...")
    df_logs = pd.read_csv(logs_p)
    print(f"資料筆數: {len(df_logs):,} 筆")

    generate_phys_benchmark_table(df_logs, Path(args.out_table))
    plot_rock_physics_template(df_logs, Path(args.dataset_csv), Path(args.out_rpt))
    compute_and_plot_walther_matrix(Path(args.strata_csv), Path(args.out_walther))


if __name__ == "__main__":
    main()
