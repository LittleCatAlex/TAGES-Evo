"""Generate 3D Voxel Seismic Microzonation Maps for All Six Taiwan Basins.

Basins included:
1. 嘉南平原 (Chianan Foreland Basin)
2. 濁水溪沖積扇 (Zhuoshui Alluvial Fan)
3. 臺北盆地 (Taipei Basin)
4. 臺中盆地 (Taichung Basin)
5. 屏東平原 (Pingtung Plain)
6. 蘭陽平原 (Yilan Plain)

Outputs:
- Individual 2-panel maps (f0 & peak amplification) for each basin:
  reports/{basin}_microzonation_map.png (300 DPI)
- Unified metrics JSON: reports/taiwan_all_basins_microzonation_metrics.json
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tages.seismic.wave_propagator import (
    compute_haskell_transfer_function,
    build_voxel_1d_stratified_profile,
)

BASIN_CONFIGS = {
    "chianan": {
        "name_zh": "嘉南前陸盆地",
        "name_en": "Chianan Foreland Basin",
        "prefix": "chianan",
        "csv_key": "Chianan",
        "structural_labels": [
            {"text": "北港基盤高區\n(淺基盤/高頻)", "lon": 120.25, "lat": 23.58, "color": "#FFE66D"},
            {"text": "臺南前陸深槽\n(厚沉積/長週期)", "lon": 120.22, "lat": 23.22, "color": "#FFE66D"}
        ]
    },
    "zhuoshui": {
        "name_zh": "濁水溪沖積扇",
        "name_en": "Zhuoshui Alluvial Fan",
        "prefix": "zhuoshui",
        "csv_key": "Zhuoshui",
        "structural_labels": [
            {"text": "扇頂出谷口 (竹山/二水)\n(巨厚礫石/高基頻)", "lon": 120.61, "lat": 23.80, "color": "#FFE66D"},
            {"text": "扇尾海口端 (麥寮/臺西)\n(濱海細泥/強烈放大)", "lon": 120.20, "lat": 23.75, "color": "#FFE66D"}
        ]
    },
    "taipei": {
        "name_zh": "臺北盆地",
        "name_en": "Taipei Basin",
        "prefix": "taipei",
        "csv_key": "Taipei",
        "structural_labels": [
            {"text": "山腳斷層下陷深槽 (五股/新莊)\n(松山層軟泥極端增厚)", "lon": 121.45, "lat": 25.07, "color": "#FFE66D"},
            {"text": "盆地東南淺基盤 (信義/大安)\n(基盤抬升/中高頻)", "lon": 121.56, "lat": 25.03, "color": "#FFE66D"}
        ]
    },
    "taichung": {
        "name_zh": "臺中盆地",
        "name_en": "Taichung Basin",
        "prefix": "taipei",  # placeholder, overwritten below
        "csv_key": "Taichung",
        "structural_labels": [
            {"text": "車籠埔斷層上盤 (東側山麓)\n(構造抬升/堅硬岩盤)", "lon": 120.73, "lat": 24.18, "color": "#FFE66D"},
            {"text": "大肚台地與市中心沖積層\n(厚層卵礫石台地)", "lon": 120.60, "lat": 24.16, "color": "#FFE66D"}
        ]
    },
    "pingtung": {
        "name_zh": "屏東平原",
        "name_en": "Pingtung Plain",
        "prefix": "pingtung",
        "csv_key": "Pingtung",
        "structural_labels": [
            {"text": "潮州斷層西側前陸深槽\n(東港/林邊深厚沉積)", "lon": 120.48, "lat": 22.50, "color": "#FFE66D"},
            {"text": "北部旗山/美濃山麓帶\n(礫石洪積扇/高基頻)", "lon": 120.55, "lat": 22.82, "color": "#FFE66D"}
        ]
    },
    "yilan": {
        "name_zh": "蘭陽平原",
        "name_en": "Yilan Plain",
        "prefix": "yilan",
        "csv_key": "Yilan",
        "structural_labels": [
            {"text": "沖繩海槽張裂沉積中心 (壯圍/五結)\n(超軟弱海相泥相)", "lon": 121.82, "lat": 24.73, "color": "#FFE66D"},
            {"text": "平原頂部沖積扇 (員山/礁溪)\n(山麓粗砂礫/淺基盤)", "lon": 121.72, "lat": 24.81, "color": "#FFE66D"}
        ]
    }
}
BASIN_CONFIGS["taichung"]["prefix"] = "taichung"


def main():
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    dense_npz_path = ROOT / "data" / "processed" / "taiwan_basins_dense.npz"
    summary_csv_path = ROOT / "reports" / "tsmip_closed_loop_summary.csv"

    if not dense_npz_path.exists():
        print(f"Error: {dense_npz_path} not found.")
        sys.exit(1)

    print("Loading 3D Voxel dataset:", dense_npz_path)
    vol = np.load(dense_npz_path)
    df_res = pd.read_csv(summary_csv_path) if summary_csv_path.exists() else pd.DataFrame()

    all_metrics = {}

    for b_key, b_cfg in BASIN_CONFIGS.items():
        prefix = b_cfg["prefix"]
        name_zh = b_cfg["name_zh"]
        name_en = b_cfg["name_en"]
        print(f"\nProcessing Basin [{name_zh}] ({name_en})...")

        lats = vol[f"{prefix}_lats"]
        lons = vol[f"{prefix}_lons"]
        lith = vol[f"{prefix}_lith"]
        z_eng = vol[f"{prefix}_bedrock_eng"]
        z_ac = vol[f"{prefix}_bedrock_ac"]
        depths = vol[f"{prefix}_depths"]

        n_lat, n_lon = len(lats), len(lons)
        print(f"  Grid dimension: {n_lat} x {n_lon} = {n_lat * n_lon} voxel columns")

        f0_grid = np.zeros((n_lat, n_lon))
        amp_grid = np.zeros((n_lat, n_lon))
        h_eff_grid = np.zeros((n_lat, n_lon))
        vs30_grid = np.zeros((n_lat, n_lon))

        for i in range(n_lat):
            for j in range(n_lon):
                prof = build_voxel_1d_stratified_profile(
                    depths=depths,
                    lith_col=lith[i, j, :],
                    z_eng=float(z_eng[i, j]),
                    z_ac=float(z_ac[i, j]),
                    basin_key=b_key
                )
                hk = compute_haskell_transfer_function(
                    thicknesses=prof["thicks"],
                    vs=prof["vs_layers"],
                    densities=prof["rho_layers"],
                    qs=prof["qs_layers"],
                    vs_bedrock=760.0,
                    density_bedrock=2.35,
                    qs_bedrock=65.0,
                    freq_range=(0.2, 5.0),
                    n_freqs=120
                )
                pks, _ = find_peaks(hk["amplification"], height=1.3, prominence=0.15, distance=5)
                f0_val = float(hk["freqs"][pks[0]]) if len(pks) > 0 else float(hk["f0"])
                amp_val = float(hk["amplification"][pks[0]]) if len(pks) > 0 else float(hk["peak_amp"])

                f0_grid[i, j] = f0_val
                amp_grid[i, j] = amp_val
                h_eff_grid[i, j] = prof["h_eff_voxel_m"]
                vs30_grid[i, j] = prof["vs30_voxel_ms"]

        # Calculate metrics
        b_metrics = {
            "n_columns": int(n_lat * n_lon),
            "f0_hz": {
                "min": float(np.min(f0_grid)),
                "max": float(np.max(f0_grid)),
                "mean": float(np.mean(f0_grid)),
                "std": float(np.std(f0_grid))
            },
            "period_t0_s": {
                "min": float(1.0 / max(np.max(f0_grid), 0.1)),
                "max": float(1.0 / max(np.min(f0_grid), 0.1)),
                "mean": float(np.mean(1.0 / f0_grid)),
            },
            "peak_amplification": {
                "min": float(np.min(amp_grid)),
                "max": float(np.max(amp_grid)),
                "mean": float(np.mean(amp_grid)),
                "std": float(np.std(amp_grid))
            },
            "h_eff_m": {
                "mean": float(np.mean(h_eff_grid)),
                "min": float(np.min(h_eff_grid)),
                "max": float(np.max(h_eff_grid)),
            },
            "vs30_ms": {
                "mean": float(np.mean(vs30_grid)),
                "min": float(np.min(vs30_grid)),
                "max": float(np.max(vs30_grid)),
            }
        }
        all_metrics[b_key] = b_metrics

        print(f"  f0 range: {b_metrics['f0_hz']['min']:.2f} ~ {b_metrics['f0_hz']['max']:.2f} Hz (mean {b_metrics['f0_hz']['mean']:.2f} Hz)")
        print(f"  Peak Amp range: {b_metrics['peak_amplification']['min']:.2f} ~ {b_metrics['peak_amplification']['max']:.2f} (mean {b_metrics['peak_amplification']['mean']:.2f})")

        # Plot 2-panel microzonation map
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7.5))
        lon_m, lat_m = np.meshgrid(lons, lats)

        # (a) Resonance Frequency f0
        c1 = ax_a.contourf(lon_m, lat_m, f0_grid, levels=20, cmap="viridis_r")
        contour_levels = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        valid_c_levels = [lv for lv in contour_levels if np.min(f0_grid) < lv < np.max(f0_grid)]
        if valid_c_levels:
            cs1 = ax_a.contour(lon_m, lat_m, f0_grid, levels=valid_c_levels, colors="black", linewidths=0.8, linestyles="--")
            ax_a.clabel(cs1, inline=True, fontsize=8, fmt="%.1f Hz")
        cb1 = fig.colorbar(c1, ax=ax_a, orientation="vertical", pad=0.02)
        cb1.set_label("共振基頻 Resonance Frequency f0 (Hz)", fontsize=10.5, fontweight="bold")

        # (b) Peak Amplification A_max
        c2 = ax_b.contourf(lon_m, lat_m, amp_grid, levels=20, cmap="magma")
        amp_levels = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
        valid_a_levels = [lv for lv in amp_levels if np.min(amp_grid) < lv < np.max(amp_grid)]
        if valid_a_levels:
            cs2 = ax_b.contour(lon_m, lat_m, amp_grid, levels=valid_a_levels, colors="white", linewidths=0.8, linestyles="--")
            ax_b.clabel(cs2, inline=True, fontsize=8, fmt="%.1f")
        cb2 = fig.colorbar(c2, ax=ax_b, orientation="vertical", pad=0.02)
        cb2.set_label("峰值放大倍率 Peak Amplification", fontsize=10.5, fontweight="bold")

        # Overlay TSMIP Stations if in basin
        df_b = df_res[df_res["basin"] == b_cfg["csv_key"]] if not df_res.empty else pd.DataFrame()
        if not df_b.empty:
            ax_a.scatter(df_b["lon"], df_b["lat"], color="white", edgecolor="black", s=70, marker="^", label="TSMIP 強震測站", zorder=5)
            for _, r in df_b.iterrows():
                ax_a.annotate(f"{r['station_code']}\n({r['f0_obs_hz']:.2f}Hz)", xy=(r["lon"], r["lat"]), xytext=(4, 4),
                              textcoords="offset points", fontsize=8, fontweight="bold", color="white",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"))

            ax_b.scatter(df_b["lon"], df_b["lat"], color="cyan", edgecolor="black", s=70, marker="^", label="TSMIP 強震測站", zorder=5)
            for _, r in df_b.iterrows():
                amp_txt = f"{r['peak_amp_obs']:.1f}" if "peak_amp_obs" in r and pd.notna(r["peak_amp_obs"]) else f"{r['f0_voxel_hz']:.2f}Hz"
                ax_b.annotate(f"{r['station_code']}\n({amp_txt})", xy=(r["lon"], r["lat"]), xytext=(4, 4),
                              textcoords="offset points", fontsize=8, fontweight="bold", color="black",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

        # Structural annotations
        for slab in b_cfg["structural_labels"]:
            ax_a.text(slab["lon"], slab["lat"], slab["text"], fontsize=9.2, fontweight="bold", color=slab["color"],
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.75))

        ax_a.set_title(f"(a) {name_zh} 強震共振基頻 (f0) 空間微區劃", fontsize=12, fontweight="bold")
        ax_a.set_xlabel("經度 Longitude (°E)", fontsize=11, fontweight="bold")
        ax_a.set_ylabel("緯度 Latitude (°N)", fontsize=11, fontweight="bold")
        if not df_b.empty:
            ax_a.legend(loc="lower left", fontsize=9.5, frameon=True)
        ax_a.grid(True, linestyle=":", alpha=0.3)

        ax_b.set_title(f"(b) {name_zh} 場址峰值放大倍率 (Peak Amplification) 空間微區劃", fontsize=12, fontweight="bold")
        ax_b.set_xlabel("經度 Longitude (°E)", fontsize=11, fontweight="bold")
        ax_b.set_ylabel("緯度 Latitude (°N)", fontsize=11, fontweight="bold")
        if not df_b.empty:
            ax_b.legend(loc="lower left", fontsize=9.5, frameon=True)
        ax_b.grid(True, linestyle=":", alpha=0.3)

        plt.suptitle(f"{name_zh} 強震長週期共振主頻與場址放大倍率三維體素空間微區劃地圖\n3D Voxel Seismic Microzonation Map of {name_en}", fontsize=13.5, fontweight="bold", y=0.99)
        plt.tight_layout()
        out_fig_path = reports_dir / f"{prefix}_microzonation_map.png"
        fig.savefig(out_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {out_fig_path}")

    # Save metrics JSON
    metrics_path = reports_dir / "taiwan_all_basins_microzonation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nAll Basin Microzonation Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
