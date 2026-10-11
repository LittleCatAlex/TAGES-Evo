"""全域 3D 地下體素地體模型生成與多維地質切片繪圖腳本 (Full 3D Voxel Generator & Multi-Slice Visualizer)。

目標深度：0 ~ 1,000 公尺 (涵蓋淺層鑽孔檢驗帶 0~300m 與深部構造基盤帶 300~1000m)。
範圍：嘉義沿海至阿里山麓山帶 (120.10°E ~ 120.60°E, 23.35°N ~ 23.60°N)。

產出：
1. data/processed/chiayi_3d_voxels_1km.npz
2. reports/chiayi_depth_slices_1km.png (六個代表性深度水平均質切片)
3. reports/chiayi_bedrock_depth_map.png (嘉義全區聲學與工程基盤等深線圖)
4. reports/chiayi_ortho_fence_diagram.png (正交立體柵欄剖面圖)
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.voxels.volume import VoxelGridConfig, VoxelVolumeGenerator

# 標準地質色表與名稱 (L2 10 類地體架構)
LITH_SPEC = {
    1: {"color": "#D2B48C", "name": "Topsoil / Fill"},
    2: {"color": "#8B4513", "name": "Clay / Mud"},
    3: {"color": "#CD853F", "name": "Silt"},
    4: {"color": "#FFD700", "name": "Fine Sand"},
    5: {"color": "#DAA520", "name": "Coarse Sand"},
    6: {"color": "#B8860B", "name": "Sand-Mud Mixed"},
    7: {"color": "#708090", "name": "Gravel"},
    8: {"color": "#4682B4", "name": "Mudstone / Shale"},
    9: {"color": "#2E8B57", "name": "Sandstone"},
    10: {"color": "#1C3144", "name": "Crystalline Basement"}
}

def get_lith_colormap():
    """建立對應 1~10 類別之離散 Colormap。"""
    colors = ["#FFFFFF"] * 11
    for lid, meta in LITH_SPEC.items():
        colors[lid] = meta["color"]
    return ListedColormap(colors)

def plot_depth_slices(data: dict, out_png: Path):
    """繪製 6 個不同深度之水平地質分布切片圖。"""
    lats = data["lats"]
    lons = data["lons"]
    depths = data["depths"]
    lith_3d = data["lithology_3d"]
    cmap = get_lith_colormap()

    target_depths = [20.0, 100.0, 250.0, 500.0, 750.0, 1000.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, tz in enumerate(target_depths):
        ax = axes[idx]
        z_idx = np.argmin(np.abs(depths - tz))
        actual_z = depths[z_idx]
        slice_lith = lith_3d[:, :, z_idx]  # (n_lat, n_lon)

        im = ax.imshow(
            slice_lith,
            origin="lower",
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=cmap,
            vmin=0,
            vmax=10,
            aspect="auto"
        )
        ax.set_title(f"Depth = {actual_z:.0f} m ({'Calibrated' if actual_z <= 300 else 'Geophysical Extrapolated'})", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("Longitude (°E)", fontsize=11)

    # 圖例
    legend_handles = [
        mpatches.Patch(color=meta["color"], label=f"{meta['name']}")
        for lid, meta in LITH_SPEC.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Taiwan 3D Subsurface Lithology: Horizontal Depth Slices across Chiayi (0 - 1,000 m)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"深度切片圖已輸出至: {out_png}")

def plot_bedrock_map(data: dict, out_png: Path):
    """繪製聲學基盤深度等深線圖 (Bedrock Surface Depth Map)。"""
    lats = data["lats"]
    lons = data["lons"]
    bedrock_2d = data["bedrock_depth_2d"]

    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(lons, lats)

    # 色階與等高線
    levels = np.linspace(50, 1000, 20)
    cf = ax.contourf(X, Y, bedrock_2d, levels=levels, cmap="magma_r", extend="both")
    cs = ax.contour(X, Y, bedrock_2d, levels=levels[::2], colors="black", linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fmt="%.0fm", fontsize=9)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label("Acoustic Bedrock Depth (m) [Vs >= 1,500 m/s]", fontsize=11)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Chiayi Acoustic Bedrock Depth Map (Z_bedrock)\nRevealing Deep Basin Depocenter & Foothill Uplift", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 標記重要地理位置
    landmarks = [
        ("Dongshi (Coast)", 120.15, 23.45),
        ("Chiayi City", 120.45, 23.48),
        ("Chukou Fault", 120.57, 23.44)
    ]
    for name, lx, ly in landmarks:
        ax.plot(lx, ly, marker="^", color="cyan", markersize=8, markeredgecolor="black")
        ax.text(lx + 0.01, ly, name, fontsize=10, fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"基盤深度圖已輸出至: {out_png}")

def plot_fence_diagram(data: dict, out_png: Path):
    """繪製正交東西向與南北向立體切片圖。"""
    lats = data["lats"]
    lons = data["lons"]
    depths = data["depths"]
    lith_3d = data["lithology_3d"]
    cmap = get_lith_colormap()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 東西向剖面 (沿 Lat = 23.48°N，穿過嘉義市中心)
    lat_idx = np.argmin(np.abs(lats - 23.48))
    ew_slice = lith_3d[lat_idx, :, :].T  # (n_depth, n_lon)

    ax1.imshow(
        ew_slice,
        extent=[lons.min(), lons.max(), depths.max(), depths.min()],
        cmap=cmap,
        vmin=0,
        vmax=10,
        aspect="auto"
    )
    ax1.set_title(f"A) East-West Cross Section along Lat {lats[lat_idx]:.2f}°N (Passing Chiayi City)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=11)
    ax1.set_ylabel("Depth (m)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. 南北向剖面 (沿 Lon = 120.35°E，貫穿嘉南平原)
    lon_idx = np.argmin(np.abs(lons - 120.35))
    ns_slice = lith_3d[:, lon_idx, :].T  # (n_depth, n_lat)

    ax2.imshow(
        ns_slice,
        extent=[lats.min(), lats.max(), depths.max(), depths.min()],
        cmap=cmap,
        vmin=0,
        vmax=10,
        aspect="auto"
    )
    ax2.set_title(f"B) North-South Cross Section along Lon {lons[lon_idx]:.2f}°E (Chiayi Plain Axis)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latitude (°N)", fontsize=11)
    ax2.set_ylabel("Depth (m)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(color=meta["color"], label=f"{meta['name']}")
        for lid, meta in LITH_SPEC.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"立體剖面圖已輸出至: {out_png}")

def main():
    root = Path(__file__).resolve().parents[1]
    vel_csv = root / "data" / "interim" / "chiayi_velocity_3d.csv"
    model_pkl = root / "models" / "tages_rf_augmented.pkl"
    out_npz = root / "data" / "processed" / "chiayi_3d_voxels_1km.npz"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("啟動嘉義全域 3D 地下地體結構體素生成 (0 ~ 1,000 m)...")
    print("==================================================")

    cfg = VoxelGridConfig(
        lon_min=120.10, lon_max=120.60, n_lon=51,
        lat_min=23.35, lat_max=23.60, n_lat=26,
        depth_min=0.0, depth_max=1000.0, n_depth=101
    )

    generator = VoxelVolumeGenerator(vel_csv, model_pkl, config=cfg)
    data = generator.generate()

    print(f"體素計算完成: 形狀 {data['lithology_3d'].shape}，共 {data['lithology_3d'].size:,} 個空間體素點。")
    generator.save_volume(data, out_npz)

    # 統計地質分佈
    unique, counts = np.unique(data["lithology_3d"], return_counts=True)
    total_voxels = data["lithology_3d"].size
    print("\n[3D 體素總體地質體積統計 (0 ~ 1,000m)]:")
    for u, c in zip(unique, counts):
        name = LITH_SPEC.get(u, {}).get("name", f"Class {u}")
        print(f"  {name:25s}: {c:6d} voxels ({c / total_voxels * 100:5.2f}%)")

    # 繪製視覺化圖表
    print("\n繪製出版級多維地質圖件...")
    plot_depth_slices(data, reports_dir / "chiayi_depth_slices_1km.png")
    plot_bedrock_map(data, reports_dir / "chiayi_bedrock_depth_map.png")
    plot_fence_diagram(data, reports_dir / "chiayi_ortho_fence_diagram.png")

    print("\n所有 3D 體素資料與地質圖件產出完畢！")

if __name__ == "__main__":
    main()
