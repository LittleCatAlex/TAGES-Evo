"""全臺灣各大盆地地質結構對比與觀測網分佈繪圖腳本 (Taiwan Regional Geological Summary & Basin Comparison Plotter)。

產出：
1. reports/taiwan_wells_and_faults_map.png (全島 423 口深層水文地質觀測井與 9 大活動斷層分佈圖)
2. reports/taiwan_basin_bedrock_comparison.png (全島六大代表性盆地之基盤深度與波速對比柱狀/小提琴圖)
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_wells_and_faults(df_bh: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(10, 12))

    # 1. 繪製六大代表性盆地範圍方框
    basin_boxes = {
        "Taipei Basin (臺北盆地)": [121.42, 24.98, 121.62, 25.12, "#ff0077"],
        "Taichung Basin (臺中盆地)": [120.58, 24.08, 120.78, 24.28, "#ffaa00"],
        "Zhuoshui Fan (濁水溪沖積扇)": [120.15, 23.65, 120.65, 23.90, "#00ddaa"],
        "Chianan Plain (嘉南平原)": [120.08, 23.15, 120.60, 23.65, "#3388ff"],
        "Pingtung Plain (屏東平原)": [120.42, 22.48, 120.68, 22.78, "#ff5533"],
        "Lanyang Plain (蘭陽平原)": [121.68, 24.65, 121.88, 24.85, "#aa00ff"]
    }
    for b_name, (x1, y1, x2, y2, col) in basin_boxes.items():
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, facecolor=col, alpha=0.15, edgecolor=col, linewidth=2, linestyle="--")
        ax.add_patch(rect)
        ax.text(x1 + 0.02, y2 - 0.03, b_name, color=col, fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # 2. 繪製 423 口深井 (依深度色彩映射)
    sc = ax.scatter(
        df_bh["lon"], df_bh["lat"],
        c=df_bh["well_depth_m"],
        cmap="turbo",
        s=45,
        alpha=0.88,
        edgecolors="black",
        linewidths=0.5,
        zorder=5
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Drilled Well Depth (m) [423 Stations across Taiwan]", fontsize=11)

    # 3. 繪製活動斷層跡線
    fault_coords = [
        ("Chelungpu Fault (車籠埔斷層)", [[120.73, 24.32], [120.71, 24.13], [120.69, 23.94], [120.68, 23.82]], "red"),
        ("Chukou Fault (觸口斷層)", [[120.59, 23.58], [120.56, 23.46], [120.53, 23.28]], "crimson"),
        ("Meishan Fault (梅山斷層)", [[120.60, 23.58], [120.48, 23.56], [120.38, 23.55]], "darkorange"),
        ("Shanchiao Fault (山腳斷層)", [[121.41, 25.04], [121.47, 25.13], [121.58, 25.24]], "deeppink"),
        ("Chaochou Fault (潮州斷層)", [[120.63, 22.82], [120.61, 22.61], [120.61, 22.40]], "firebrick"),
        ("Chihshang Fault (池上斷層)", [[121.32, 23.28], [121.22, 23.08], [121.14, 22.88]], "cyan")
    ]
    for fname, coords, fcol in fault_coords:
        pts = np.array(coords)
        ax.plot(pts[:, 0], pts[:, 1], color=fcol, linewidth=2.8, alpha=0.9, zorder=4)

    ax.set_xlim(119.8, 122.2)
    ax.set_ylim(21.8, 25.4)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Taiwan Subsurface Geological Observation Network (423 Deep Boreholes & Active Faults)", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"全島觀測網圖已儲存至: {out_png}")

def plot_basin_comparison(npz_path: Path, out_png: Path):
    data = np.load(npz_path)
    basins = [
        ("Taipei Basin\n(臺北盆地)", "taipei", "#ff0077"),
        ("Taichung Basin\n(臺中盆地)", "taichung", "#ffaa00"),
        ("Zhuoshui Fan\n(濁水溪沖積扇)", "zhuoshui", "#00ddaa"),
        ("Chianan Plain\n(嘉南平原)", "chianan", "#3388ff"),
        ("Pingtung Plain\n(屏東平原)", "pingtung", "#ff5533"),
        ("Lanyang Plain\n(蘭陽平原)", "yilan", "#aa00ff")
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    bedrock_means = []
    bedrock_stds = []
    vs_means = []
    names = []
    colors = []

    for name, key, col in basins:
        b2d = data[f"{key}_bedrock"]
        vs3d = data[f"{key}_vs"]
        names.append(name)
        colors.append(col)
        bedrock_means.append(np.mean(b2d))
        bedrock_stds.append(np.std(b2d))
        vs_means.append(np.mean(vs3d))

    x = np.arange(len(names))
    width = 0.55

    # 1. 基盤深度對比
    bars1 = ax1.bar(x, bedrock_means, width, yerr=bedrock_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Acoustic Bedrock Depth Z_bedrock (m)", fontsize=11)
    ax1.set_title("A) Mean Acoustic Bedrock Depth Across Taiwan Basins", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars1, bedrock_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 20, f"{m:.0f}m", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # 2. 平均波速對比
    bars2 = ax2.bar(x, vs_means, width, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("Mean Shear-Wave Velocity Vs (m/s) [0 ~ 1,000m]", fontsize=11)
    ax2.set_title("B) Mean 3D Shear-Wave Velocity (TEC LEE2025) in Basins", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar, m in zip(bars2, vs_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 30, f"{m:.0f} m/s", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"全島盆地對比圖已儲存至: {out_png}")

def main():
    root = Path(__file__).resolve().parents[1]
    df_bh = pd.read_csv(root / "data" / "interim" / "taiwan_boreholes.csv")
    npz_path = root / "data" / "processed" / "taiwan_basins_dense.npz"
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    plot_wells_and_faults(df_bh, rep_dir / "taiwan_wells_and_faults_map.png")
    plot_basin_comparison(npz_path, rep_dir / "taiwan_basin_bedrock_comparison.png")

if __name__ == "__main__":
    main()
