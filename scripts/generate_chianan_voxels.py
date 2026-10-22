"""西南臺灣 (嘉南平原與雲嘉南地區) 全域 3D 地下體素地體模型生成與區域構造視覺化腳本。

範圍：
  經度：120.00°E ~ 120.75°E (臺灣海峽沿海至阿里山麓山帶前緣)
  緯度：22.95°N ~ 23.85°N (濁水溪以北雲林至二仁溪以南臺南全境)
  深度：0 ~ 1,000 公尺 (10m 垂直解析度，涵蓋 0~300m 鑽井層與 300~1000m 深部前陸基底)

產出：
1. data/processed/chianan_3d_voxels_1km.npz (西南臺灣全域 3D 體素檔)
2. reports/chianan_depth_slices_1km.png (六大水平深度地質分布切片圖)
3. reports/chianan_bedrock_depth_map.png (全區聲學基盤等深線圖，揭示北港高台與臺南盆地對比)
4. reports/chianan_regional_transects.png (南北向與東西向區域地質正交立體剖面圖)
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, tz in enumerate(target_depths):
        ax = axes[idx]
        z_idx = np.argmin(np.abs(depths - tz))
        actual_z = depths[z_idx]
        slice_lith = lith_3d[:, :, z_idx]

        im = ax.imshow(
            slice_lith,
            origin="lower",
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=cmap,
            vmin=0,
            vmax=10,
            aspect="auto"
        )
        tag = "Borehole Constrained" if actual_z <= 300 else "Geophysical / Tomographic Extrapolated"
        ax.set_title(f"Depth = {actual_z:.0f} m ({tag})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("Longitude (°E)", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=meta["color"], label=f"{meta['name']}")
        for lid, meta in LITH_SPEC.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Southwestern Taiwan (Chianan Plain) 3D Subsurface Lithology: Horizontal Depth Slices (0 - 1,000 m)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"全區深度切片圖已輸出至: {out_png}")

def plot_bedrock_map(data: dict, out_png: Path):
    """繪製區域聲學基盤深度等深線圖 (Bedrock Surface Depth Map)。"""
    lats = data["lats"]
    lons = data["lons"]
    bedrock_2d = data["bedrock_depth_2d"]

    fig, ax = plt.subplots(figsize=(11, 10))
    X, Y = np.meshgrid(lons, lats)

    levels = np.linspace(100, 1000, 19)
    cf = ax.contourf(X, Y, bedrock_2d, levels=levels, cmap="viridis_r", extend="both")
    cs = ax.contour(X, Y, bedrock_2d, levels=levels[::2], colors="black", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, inline=True, fmt="%.0fm", fontsize=9)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label("Acoustic Bedrock Depth (m) [Vs >= 2,150 m/s]", fontsize=11)

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Southwestern Taiwan Foreland Basin: Acoustic Bedrock Depth Map (Z_bedrock)\nHighlighting Peikang High vs Tainan Basin Depocenter", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 標記地質構造與城鎮地標
    landmarks = [
        ("Mailiao", 120.18, 23.75, "city"),
        ("Douliu", 120.55, 23.71, "city"),
        ("Beigang (Peikang High)", 120.30, 23.58, "struc"),
        ("Chiayi City", 120.45, 23.48, "city"),
        ("Xinying", 120.31, 23.31, "city"),
        ("Tainan City (Depocenter)", 120.21, 22.99, "struc"),
        ("Chukou Fault", 120.58, 23.44, "fault"),
        ("Meishan Fault", 120.53, 23.57, "fault")
    ]
    for name, lx, ly, kind in landmarks:
        if kind == "struc":
            ax.plot(lx, ly, marker="*", color="gold", markersize=12, markeredgecolor="black")
            ax.text(lx + 0.015, ly - 0.015, name, fontsize=10, fontweight="bold", color="yellow",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="darkblue", alpha=0.7))
        elif kind == "fault":
            ax.plot(lx, ly, marker="X", color="red", markersize=9, markeredgecolor="black")
            ax.text(lx + 0.012, ly + 0.012, name, fontsize=9, fontweight="bold", color="salmon",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
        else:
            ax.plot(lx, ly, marker="o", color="cyan", markersize=7, markeredgecolor="black")
            ax.text(lx + 0.012, ly, name, fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"全區基盤深度圖已輸出至: {out_png}")

def plot_regional_transects(data: dict, out_png: Path):
    """繪製南北向 (Yunlin -> Chiayi -> Tainan) 與東西向區域地質正交立體剖面圖。"""
    lats = data["lats"]
    lons = data["lons"]
    depths = data["depths"]
    lith_3d = data["lithology_3d"]
    cmap = get_lith_colormap()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11))

    # 1. 南北向剖面 (沿 Lon = 120.30°E 貫穿北港高台與臺南盆地)
    lon_idx = np.argmin(np.abs(lons - 120.30))
    ns_slice = lith_3d[:, lon_idx, :].T  # (n_depth, n_lat)

    im1 = ax1.imshow(
        ns_slice,
        extent=[lats.min(), lats.max(), depths.max(), depths.min()],
        cmap=cmap,
        vmin=0,
        vmax=10,
        aspect="auto"
    )
    ax1.set_title(f"A) North-South Regional Transect along Lon {lons[lon_idx]:.2f}°E (Peikang High -> Tainan Depocenter)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Latitude (°N) [South: Tainan Basin <---> North: Yunlin / Zhuoshui River]", fontsize=11)
    ax1.set_ylabel("Depth (m)", fontsize=11)
    ax1.axvline(23.58, color="white", linestyle="--", alpha=0.8, label="Peikang High Axis")
    ax1.axvline(23.31, color="cyan", linestyle=":", alpha=0.8, label="Xinying / Bazhang River")
    ax1.legend(loc="upper right", framealpha=0.85)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. 東西向剖面 (沿 Lat = 23.45°N 穿過嘉南平原中軸)
    lat_idx = np.argmin(np.abs(lats - 23.45))
    ew_slice = lith_3d[lat_idx, :, :].T  # (n_depth, n_lon)

    im2 = ax2.imshow(
        ew_slice,
        extent=[lons.min(), lons.max(), depths.max(), depths.min()],
        cmap=cmap,
        vmin=0,
        vmax=10,
        aspect="auto"
    )
    ax2.set_title(f"B) East-West Regional Transect along Lat {lats[lat_idx]:.2f}°N (Coast -> Chiayi Plain -> Chukou Fault Front)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude (°E) [West: Coastal Plain <---> East: Foothills / Alishan Front]", fontsize=11)
    ax2.set_ylabel("Depth (m)", fontsize=11)
    ax2.axvline(120.58, color="red", linestyle="--", alpha=0.8, label="Chukou Fault Zone")
    ax2.legend(loc="upper right", framealpha=0.85)
    ax2.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(color=meta["color"], label=f"{meta['name']}")
        for lid, meta in LITH_SPEC.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"區域正交剖面圖已輸出至: {out_png}")

def main():
    root = Path(__file__).resolve().parents[1]
    vel_csv = root / "data" / "interim" / "chianan_velocity_3d.csv"
    model_distilled = root / "models" / "well_seq_net_distilled.pt"
    model_pt = root / "models" / "well_seq_net_v1.pt"
    model_path = model_distilled if model_distilled.exists() else (model_pt if model_pt.exists() else (root / "models" / "tages_rf_augmented.pkl"))
    out_npz = root / "data" / "processed" / "chianan_3d_voxels_1km.npz"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print(f"啟動西南臺灣 (嘉南平原/雲嘉南) 全域 3D 地下體素地體模型生成 (0 ~ 1,000 m, 模型: {model_path.name})...")
    print("================================================================")

    # 擴展網格配置 (經向 120.00 ~ 120.75, 緯向 22.95 ~ 23.85, 深度 0 ~ 1000m)
    cfg = VoxelGridConfig(
        lon_min=120.00, lon_max=120.75, n_lon=46,  # 步長約 1.6 km
        lat_min=22.95, lat_max=23.85, n_lat=46,    # 步長約 2.1 km
        depth_min=0.0, depth_max=1000.0, n_depth=101 # 垂直步長 10.0 m
    )

    generator = VoxelVolumeGenerator(vel_csv, model_path, config=cfg)
    data = generator.generate()

    print(f"體素計算完成: 形狀 {data['lithology_3d'].shape}，共 {data['lithology_3d'].size:,} 個空間體素點。")
    generator.save_volume(data, out_npz)

    # 統計地質分佈
    unique, counts = np.unique(data["lithology_3d"], return_counts=True)
    total_voxels = data["lithology_3d"].size
    print("\n[西南臺灣全區 3D 體素總體地質體積統計 (0 ~ 1,000m)]:")
    for u, c in zip(unique, counts):
        name = LITH_SPEC.get(u, {}).get("name", f"Class {u}")
        print(f"  {name:25s}: {c:6d} voxels ({c / total_voxels * 100:5.2f}%)")

    # 繪製視覺化圖表
    print("\n繪製出版級多維地質圖件...")
    plot_depth_slices(data, reports_dir / "chianan_depth_slices_1km.png")
    plot_bedrock_map(data, reports_dir / "chianan_bedrock_depth_map.png")
    plot_regional_transects(data, reports_dir / "chianan_regional_transects.png")

    print("\n全區 3D 體素資料與多維地質圖件產出完畢！")

if __name__ == "__main__":
    main()
