"""臺灣主要沉積盆地強震站場址效應端到端閉環驗證 (3D Voxel Model + Haskell-Thomson vs. TSMIP 實測主頻)。

科學物理依據：
1. 直接從 data/processed/taiwan_basins_dense.npz 提取全臺 32 座代表性 TSMIP 強震站正下方之
   「3D 體素預測岩性柱 (Lithology Column)」與「三維層析剪力波速剖面 (Tomography Vs)」，
   完全不依賴測站實測 Vs30 或文獻沉積深度，實現真正的 3D 體素至 1D 波傳端到端閉環正演。
2. 明確區分兩類物理基盤面定義：
   - 工程基盤面 (Engineering Bedrock, Z_eng, Vs >= 760 m/s 或堅硬礫石/半固結岩頂界)：控制 0.5-5.0 Hz 強震場址卓越共振主頻。
   - 地殼聲學基盤面 (Seismic Acoustic Bedrock, Z_ac, Vs >= 2,000 m/s)：反映深部盆地-新第三紀/中生代結晶基底構造界面。
3. 將 3D 體素岩石物理耦合剖面代入 Haskell-Thomson 傳遞矩陣計算 f0_voxel，與 TSMIP 實測主頻 f0_obs 進行獨立盲測統計檢定 (Pearson r, R^2, RMSE)。
4. 與傳統四分之一波長粗略公式 (Quarter-Wavelength Baseline) 進行定量對照，驗證納入三維沉積相異質性與多層波阻抗結構後之預測精度提升。
5. 產出嘉南前陸盆地強震長週期共振主頻 (f0) 與場址放大峰值空間微區劃地圖 (Seismic Microzonation Map)。
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.seismic.wave_propagator import (
    compute_haskell_transfer_function,
    build_voxel_1d_stratified_profile,
    compute_quarter_wavelength_f0,
    FACIES_ROCK_PHYSICS
)
from tages.labels.lithology import L2_NAMES

BASIN_COLORS = {
    "Chianan": "#E63946",
    "Zhuoshui": "#1D3557",
    "Taipei": "#2A9D8F",
    "Taichung": "#E76F51",
    "Pingtung": "#9B5DE5",
    "Yilan": "#00BBF9"
}


def find_station_voxel_column(vol: np.lib.npyio.NpzFile, lon: float, lat: float, basin_hint: str = None) -> dict:
    """自 taiwan_basins_dense.npz 尋找距離測站 (lon, lat) 最近的 3D 體素垂直柱。"""
    basin_keys = ["chianan", "zhuoshui", "taipei", "taichung", "pingtung", "yilan"]
    if basin_hint and basin_hint.lower() in basin_keys:
        order = [basin_hint.lower()] + [b for b in basin_keys if b != basin_hint.lower()]
    else:
        order = basin_keys

    best_dist = float("inf")
    best_data = None

    for b_key in order:
        if f"{b_key}_lats" not in vol:
            continue
        lats = vol[f"{b_key}_lats"]
        lons = vol[f"{b_key}_lons"]
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))
        dist = float(np.hypot(lats[lat_idx] - lat, lons[lon_idx] - lon))
        if dist < best_dist:
            best_dist = dist
            basin_surf_vs = vol[f"{b_key}_vs"][:, :, 0].astype(float)
            vs_basin_ref = float(np.percentile(basin_surf_vs, 45))
            best_data = {
                "basin_key": b_key,
                "grid_lat": float(lats[lat_idx]),
                "grid_lon": float(lons[lon_idx]),
                "depths": vol[f"{b_key}_depths"].astype(float),
                "lith_col": vol[f"{b_key}_lith"][lat_idx, lon_idx, :].astype(int),
                "vs_tomo_col": vol[f"{b_key}_vs"][lat_idx, lon_idx, :].astype(float),
                "vs_basin_ref": vs_basin_ref,
                "z_bedrock_eng": float(vol[f"{b_key}_bedrock_eng"][lat_idx, lon_idx]),
                "z_bedrock_ac": float(vol[f"{b_key}_bedrock_ac"][lat_idx, lon_idx])
            }
            if dist < 0.05:
                break
    return best_data


def run_site_effects_validation():
    root = Path(__file__).resolve().parents[1]
    truth_path = root / "data" / "raw" / "seismic" / "tsmip_egdt_site_truth.csv"
    if not truth_path.exists():
        truth_path = root / "data" / "raw" / "seismic" / "tsmip_site_truth.csv"

    npz_path = root / "data" / "processed" / "taiwan_basins_dense.npz"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("啟動 3D 體素模型驅動 32 座 TSMIP 強震站 1D Haskell-Thomson 波傳閉環驗證")
    print("=========================================================================")

    df_truth = pd.read_csv(truth_path)
    vol = np.load(npz_path)

    results = []
    station_transfer_functions = []

    print(f"\n針對全臺 {len(df_truth)} 座 TSMIP 強震站萃取 3D 體素岩性柱並執行 Haskell 傳遞矩陣正演:")
    print("-------------------------------------------------------------------------")

    for idx, row in df_truth.iterrows():
        code = row["station_code"]
        name = row["station_name"]
        basin = row["basin"]
        lon = float(row["lon"])
        lat = float(row["lat"])
        vs30_obs = float(row["vs30_obs_ms"])
        site_class = row["site_class"]
        f0_obs = float(row["f0_obs_hz"])
        amp_obs = float(row["peak_amp_obs"])
        lit_depth_ac = float(row["sediment_depth_lit_m"])
        lit_depth_eng = float(row["engineering_bedrock_depth_lit_m"]) if "engineering_bedrock_depth_lit_m" in row else 50.0
        z1_0_egdt = float(row["z1_0_egdt_m"]) if "z1_0_egdt_m" in row else lit_depth_ac
        drilled_depth = float(row["drilled_depth_m"]) if "drilled_depth_m" in row else 35.0
        kappa0 = float(row["kappa0_s"]) if "kappa0_s" in row else 0.06
        src_verified = str(row.get("source_verified", "yes"))
        data_src = str(row.get("source_ref", row.get("data_source", "NCREE EGDT")))

        # 1. 從 taiwan_basins_dense.npz 提取該站正下方之 3D 體素垂直柱
        voxel_col = find_station_voxel_column(vol, lon, lat, basin_hint=basin)
        prof = build_voxel_1d_stratified_profile(
            depths=voxel_col["depths"],
            lith_col=voxel_col["lith_col"],
            z_eng=voxel_col["z_bedrock_eng"],
            z_ac=voxel_col["z_bedrock_ac"],
            basin_key=voxel_col["basin_key"]
        )

        # 提取主導岩相名稱
        eng_mask = voxel_col["depths"] <= max(voxel_col["z_bedrock_eng"], 40.0)
        lith_in_eng = voxel_col["lith_col"][eng_mask]
        unique_l, counts_l = np.unique(lith_in_eng, return_counts=True)
        dom_lith = int(unique_l[np.argmax(counts_l)]) if len(unique_l) > 0 else 4
        dom_lith_name = L2_NAMES.get(dom_lith, f"Class-{dom_lith}")

        # 2. 代入 Haskell-Thomson 傳遞矩陣計算 3D 體素模型預測主頻 f0_voxel
        haskell_voxel = compute_haskell_transfer_function(
            thicknesses=prof["thicks"],
            vs=prof["vs_layers"],
            densities=prof["rho_layers"],
            qs=prof["qs_layers"],
            vs_bedrock=760.0,       # 工程基盤阻抗邊界 (Vs >= 760 m/s, NEHRP B/C boundary)
            density_bedrock=2.35,
            qs_bedrock=65.0,
            freq_range=(0.2, 8.0),
            n_freqs=500
        )

        peaks, _ = find_peaks(haskell_voxel["amplification"], height=1.3, prominence=0.15, distance=12)
        if len(peaks) > 0:
            f0_voxel = float(haskell_voxel["freqs"][peaks[0]])
            amp_voxel = float(haskell_voxel["amplification"][peaks[0]])
        else:
            f0_voxel = float(haskell_voxel["f0"])
            amp_voxel = float(haskell_voxel["peak_amp"])

        # 3. 四分之一波長基準公式
        f0_quarter = compute_quarter_wavelength_f0(vs30_obs, lit_depth_eng)

        err_voxel_pct = abs(f0_voxel - f0_obs) / f0_obs * 100.0
        err_quarter_pct = abs(f0_quarter - f0_obs) / f0_obs * 100.0

        print(f"[{idx+1:02d}/{len(df_truth)}] {code} ({basin} - {name}): "
              f"體素主相: {dom_lith_name} | Z_ac: {voxel_col['z_bedrock_ac']:.0f}m | "
              f"H_eff: {prof['h_eff_voxel_m']:.1f}m | Vs_avg: {prof['vs_avg_voxel_ms']:.0f}m/s | "
              f"f0_voxel: {f0_voxel:.2f}Hz vs f0_obs: {f0_obs:.2f}Hz (誤差: {err_voxel_pct:.1f}%) | "
              f"f0_quarter: {f0_quarter:.2f}Hz (誤差: {err_quarter_pct:.1f}%)")

        results.append({
            "station_code": code,
            "station_name": name,
            "basin": basin,
            "lon": lon,
            "lat": lat,
            "site_class": site_class,
            "voxel_dom_lith": dom_lith_name,
            "z_bedrock_eng_m": round(voxel_col["z_bedrock_eng"], 1),
            "z_bedrock_ac_m": round(voxel_col["z_bedrock_ac"], 1),
            "sediment_depth_lit_m": round(lit_depth_ac, 1),
            "engineering_bedrock_depth_lit_m": round(lit_depth_eng, 1),
            "sediment_depth_m": prof["h_eff_voxel_m"],
            "vs30_obs_ms": round(vs30_obs, 1),
            "vs30_voxel_ms": prof["vs30_voxel_ms"],
            "avg_vs_ms": prof["vs_avg_voxel_ms"],
            "f0_voxel_hz": round(f0_voxel, 2),
            "f0_model_hz": round(f0_voxel, 2),
            "f0_quarter_hz": round(f0_quarter, 2),
            "f0_obs_hz": round(f0_obs, 2),
            "peak_amp_model": round(amp_voxel, 2),
            "peak_amp_obs": round(amp_obs, 2),
            "rel_error_pct": round(err_voxel_pct, 1),
            "rel_error_quarter_pct": round(err_quarter_pct, 1),
            "z1_0_egdt_m": round(z1_0_egdt, 1),
            "drilled_depth_m": round(drilled_depth, 1),
            "kappa0_s": round(kappa0, 4),
            "source_verified": src_verified,
            "source_ref": data_src
        })

        station_transfer_functions.append({
            "code": code,
            "name": name,
            "basin": basin,
            "dom_lith": dom_lith_name,
            "freqs": haskell_voxel["freqs"],
            "amp": haskell_voxel["amplification"],
            "f0_voxel": f0_voxel,
            "f0_obs": f0_obs
        })

    df_res = pd.DataFrame(results)

    # 儲存 CSV 報表
    csv_out1 = reports_dir / "chianan_ssr_summary.csv"
    csv_out2 = reports_dir / "tsmip_closed_loop_summary.csv"
    df_res.to_csv(csv_out1, index=False, encoding="utf-8-sig")
    df_res.to_csv(csv_out2, index=False, encoding="utf-8-sig")
    print(f"\n場址驗證統計摘要表已儲存至: {csv_out1} 與 {csv_out2}")

    from scipy.stats import spearmanr
    # 計算統計指標 (R^2, Pearson r, Spearman rho, RMSE, MAE, MAPE)
    y_true = df_res["f0_obs_hz"].values
    y_voxel = df_res["f0_voxel_hz"].values
    y_quarter = df_res["f0_quarter_hz"].values

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    # 3D 體素 Haskell 模型指標
    rmse_voxel = float(np.sqrt(np.mean((y_true - y_voxel) ** 2)))
    mae_voxel = float(np.mean(np.abs(y_true - y_voxel)))
    ss_res_voxel = float(np.sum((y_true - y_voxel) ** 2))
    r2_voxel = float(1.0 - (ss_res_voxel / ss_tot))
    r_voxel = float(np.corrcoef(y_true, y_voxel)[0, 1])
    rho_voxel = float(spearmanr(y_true, y_voxel).statistic)
    mape_voxel = float(df_res["rel_error_pct"].mean())

    # Quarter-wavelength 基準指標
    rmse_quarter = float(np.sqrt(np.mean((y_true - y_quarter) ** 2)))
    mae_quarter = float(np.mean(np.abs(y_true - y_quarter)))
    ss_res_quarter = float(np.sum((y_true - y_quarter) ** 2))
    r2_quarter = float(1.0 - (ss_res_quarter / ss_tot))
    r_quarter = float(np.corrcoef(y_true, y_quarter)[0, 1])
    rho_quarter = float(spearmanr(y_true, y_quarter).statistic)
    mape_quarter = float(df_res["rel_error_quarter_pct"].mean())

    # 聲學基盤深度指標
    z_ac_arr = df_res["z_bedrock_ac_m"].values
    z_lit_arr = df_res["sediment_depth_lit_m"].values
    r_depth = float(np.corrcoef(z_ac_arr, z_lit_arr)[0, 1])
    rmse_depth = float(np.sqrt(np.mean((z_ac_arr - z_lit_arr) ** 2)))

    # 計算各盆地殘差偏差 (f0_voxel - f0_obs)
    basin_bias = {}
    for b_name, grp in df_res.groupby("basin"):
        bias_val = float(np.mean(grp["f0_voxel_hz"] - grp["f0_obs_hz"]))
        rmse_b = float(np.sqrt(np.mean((grp["f0_voxel_hz"] - grp["f0_obs_hz"]) ** 2)))
        basin_bias[b_name] = {
            "n_stations": int(len(grp)),
            "mean_bias_hz": round(bias_val, 3),
            "rmse_hz": round(rmse_b, 3)
        }

    metrics_summary = {
        "n_stations": len(df_res),
        "f0_voxel_vs_obs": {
            "pearson_r": round(r_voxel, 4),
            "spearman_rho": round(rho_voxel, 4),
            "r2": round(r2_voxel, 4),
            "rmse_hz": round(rmse_voxel, 3),
            "mae_hz": round(mae_voxel, 3),
            "mean_rel_error_pct": round(mape_voxel, 2)
        },
        "quarter_wavelength_vs_obs": {
            "pearson_r": round(r_quarter, 4),
            "spearman_rho": round(rho_quarter, 4),
            "r2": round(r2_quarter, 4),
            "rmse_hz": round(rmse_quarter, 3),
            "mae_hz": round(mae_quarter, 3),
            "mean_rel_error_pct": round(mape_quarter, 2)
        },
        "acoustic_bedrock_vs_lit_depth": {
            "pearson_r": round(r_depth, 4),
            "rmse_m": round(rmse_depth, 1)
        },
        "basin_residual_summary": basin_bias
    }
    with open(reports_dir / "ssr_closed_loop_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    with open(reports_dir / "tsmip_closed_loop_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    print("\n【3D 體素模型驅動 1D 波傳閉環外部獨立盲測統計檢定成果 (N=32)】:")
    print(f"  [3D 體素 Haskell 傳遞模型]")
    print(f"    皮爾森相關係數 Pearson r: {r_voxel:.4f}")
    print(f"    決定係數 R^2:              {r2_voxel:.4f}")
    print(f"    均方根誤差 RMSE:           {rmse_voxel:.3f} Hz")
    print(f"    平均相對誤差 MAPE:         {mape_voxel:.1f}%")
    print(f"  [四分之一波長粗略公式 Quarter-Wavelength Baseline]")
    print(f"    皮爾森相關係數 Pearson r: {r_quarter:.4f}")
    print(f"    決定係數 R^2:              {r2_quarter:.4f}")
    print(f"    均方根誤差 RMSE:           {rmse_quarter:.3f} Hz")
    print(f"    平均相對誤差 MAPE:         {mape_quarter:.1f}%")
    print(f"  [聲學基盤面 Z_ac vs 文獻基盤深度]")
    print(f"    相關係數 Pearson r:        {r_depth:.4f} (RMSE: {rmse_depth:.1f} m)")

    # 4. 繪製圖件 1: 六大代表性強震站 3D 體素 Haskell 頻率響應曲線與實測對照圖 (圖 D-1)
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes_flat = axes1.flatten()
    rep_stations = ["CHY054", "CHY015", "CHY002", "TAP001", "TCU054", "ILA004"]

    plot_count = 0
    for code_target in rep_stations:
        st_info = next((s for s in station_transfer_functions if s["code"] == code_target), None)
        if st_info and plot_count < 6:
            ax = axes_flat[plot_count]
            c = BASIN_COLORS.get(st_info["basin"], "#333333")
            ax.plot(st_info["freqs"], st_info["amp"], color=c, lw=2.4, label="3D 體素耦合 Haskell |H(f)|")
            ax.axvline(st_info["f0_voxel"], color=c, linestyle="--", lw=1.8, label=f"3D 體素預測 f0_voxel: {st_info['f0_voxel']:.2f} Hz")
            ax.axvline(st_info["f0_obs"], color="#111111", linestyle=":", lw=2.2, label=f"TSMIP 實測 f0_obs: {st_info['f0_obs']:.2f} Hz")

            ax.set_title(f"測站 {st_info['code']} ({st_info['basin']} - {st_info['name']}) [體素相: {st_info['dom_lith']}]", fontsize=11, fontweight="bold")
            ax.set_ylabel("放大倍率 Amplification", fontsize=10)
            ax.set_xlim(0.2, 8.0)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=9, frameon=True)
            plot_count += 1

    axes_flat[-1].set_xlabel("頻率 Frequency (Hz)", fontsize=11, fontweight="bold")
    axes_flat[-2].set_xlabel("頻率 Frequency (Hz)", fontsize=11, fontweight="bold")
    plt.suptitle("全臺代表性強震站 3D 體素模型驅動 Haskell-Thomson 場址響應閉環分析與實測驗證\nTheoretical Transfer Spectra |H(f)| vs. Observed TSMIP Site Resonance", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig1_path1 = reports_dir / "chianan_ssr_site_effects.png"
    fig1_path2 = reports_dir / "tsmip_haskell_vs_obs_spectra.png"
    fig1.savefig(fig1_path1, dpi=300, bbox_inches="tight")
    fig1.savefig(fig1_path2, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"場址響應曲線圖已儲存至: {fig1_path1} 與 {fig1_path2}")

    # 5. 繪製圖件 2: 全臺 32 座強震站 3D 體素預測 f0_voxel vs. 實測 f0_obs 散佈迴歸圖與粗略公式對比 (圖 D-2)
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)

    # 左子圖：3D 體素 Haskell 模型
    for basin, grp in df_res.groupby("basin"):
        c = BASIN_COLORS.get(basin, "#333333")
        ax2a.scatter(grp["f0_obs_hz"], grp["f0_voxel_hz"], color=c, s=100, edgecolors="black", linewidth=1.1, label=f"{basin} 盆地 (N={len(grp)})", zorder=4)
        for _, r in grp.iterrows():
            ax2a.annotate(r["station_code"], xy=(r["f0_obs_hz"], r["f0_voxel_hz"]), xytext=(3, 3), textcoords="offset points", fontsize=8, fontweight="bold")

    f_max = max(y_true.max(), y_voxel.max(), y_quarter.max()) * 1.15
    f_line = np.linspace(0.4, f_max, 100)
    ax2a.plot(f_line, f_line, color="#2B2D42", linestyle="-", lw=1.8, label="1:1 完全一致線", zorder=2)
    ax2a.fill_between(f_line, f_line * 0.8, f_line * 1.2, color="#8D99AE", alpha=0.2, label="±20% 統計容許區間", zorder=1)

    stats_text_a = (
        f"【本研究 3D 體素 Haskell 模型】\n"
        f"測站數 N = {len(df_res)} (涵蓋全臺六大盆地)\n"
        f"相關係數 r = {r_voxel:.4f}\n"
        f"決定係數 R^2 = {r2_voxel:.4f}\n"
        f"均方根誤差 RMSE = {rmse_voxel:.3f} Hz\n"
        f"平均相對誤差 = {mape_voxel:.1f}%"
    )
    ax2a.text(0.04, 0.70, stats_text_a, transform=ax2a.transAxes, fontsize=10.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888", alpha=0.9))

    ax2a.set_xlim(0.4, 3.2)
    ax2a.set_ylim(0.4, 3.2)
    ax2a.set_xlabel("氣象署 / 國震中心 TSMIP 實測共振基頻 f0_obs (Hz)", fontsize=11, fontweight="bold")
    ax2a.set_ylabel("3D 體素模型推導基頻 f0_voxel (Hz)", fontsize=11, fontweight="bold")
    ax2a.set_title("(a) 3D 體素多層介質波傳模型 (本研究閉環正演)", fontsize=12, fontweight="bold")
    ax2a.grid(True, linestyle="--", alpha=0.4)
    ax2a.legend(loc="lower right", fontsize=9, frameon=True)

    # 右子圖：四分之一波長粗略公式
    for basin, grp in df_res.groupby("basin"):
        c = BASIN_COLORS.get(basin, "#333333")
        ax2b.scatter(grp["f0_obs_hz"], grp["f0_quarter_hz"], color=c, s=100, edgecolors="black", linewidth=1.1, label=f"{basin} 盆地", zorder=4)

    ax2b.plot(f_line, f_line, color="#2B2D42", linestyle="-", lw=1.8, label="1:1 完全一致線", zorder=2)
    ax2b.fill_between(f_line, f_line * 0.8, f_line * 1.2, color="#8D99AE", alpha=0.2, label="±20% 容許區間", zorder=1)

    stats_text_b = (
        f"【四分之一波長粗略公式 (無沉積相)】\n"
        f"公式：f0 = Vs30 / (4 * H_eff)\n"
        f"相關係數 r = {r_quarter:.4f}\n"
        f"決定係數 R^2 = {r2_quarter:.4f}\n"
        f"均方根誤差 RMSE = {rmse_quarter:.3f} Hz\n"
        f"平均相對誤差 = {mape_quarter:.1f}%"
    )
    ax2b.text(0.04, 0.70, stats_text_b, transform=ax2b.transAxes, fontsize=10.5, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#FFEEBA", alpha=0.9))

    ax2b.set_xlim(0.4, 3.2)
    ax2b.set_ylim(0.4, 3.2)
    ax2b.set_xlabel("氣象署 / 國震中心 TSMIP 實測共振基頻 f0_obs (Hz)", fontsize=11, fontweight="bold")
    ax2b.set_title("(b) 傳統四分之一波長粗略公式 (Quarter-Wavelength)", fontsize=12, fontweight="bold")
    ax2b.grid(True, linestyle="--", alpha=0.4)
    ax2b.legend(loc="lower right", fontsize=9, frameon=True)

    plt.suptitle("全臺主要沉積盆地強震站 3D 體素閉環共振基頻驗證與傳統粗略模型對比\nEnd-to-End 3D Voxel Model vs. Quarter-Wavelength Baseline (N=32 Stations)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig2_path1 = reports_dir / "ssr_validation_regression.png"
    fig2_path2 = reports_dir / "tsmip_closed_loop_regression.png"
    fig2.savefig(fig2_path1, dpi=300, bbox_inches="tight")
    fig2.savefig(fig2_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"外部驗證迴歸散佈圖已儲存至: {fig2_path1} 與 {fig2_path2}")

    # 6. 繪製圖件 3: 嘉南前陸盆地強震長週期共振主頻 (f0) 與場址放大峰值空間微區劃地圖 (圖 D-3)
    print("\n正在計算嘉南盆地 1,296 個 3D 體素柱之連續空間微區劃響應...")
    lats_cn = vol["chianan_lats"]
    lons_cn = vol["chianan_lons"]
    lith_cn = vol["chianan_lith"]
    z_eng_cn = vol["chianan_bedrock_eng"]
    z_ac_cn = vol["chianan_bedrock_ac"]
    depths_cn = vol["chianan_depths"]

    f0_grid = np.zeros((len(lats_cn), len(lons_cn)))
    amp_grid = np.zeros((len(lats_cn), len(lons_cn)))

    for i in range(len(lats_cn)):
        for j in range(len(lons_cn)):
            prof_ij = build_voxel_1d_stratified_profile(
                depths=depths_cn,
                lith_col=lith_cn[i, j, :],
                z_eng=float(z_eng_cn[i, j]),
                z_ac=float(z_ac_cn[i, j]),
                basin_key="chianan"
            )
            hk_ij = compute_haskell_transfer_function(
                thicknesses=prof_ij["thicks"],
                vs=prof_ij["vs_layers"],
                densities=prof_ij["rho_layers"],
                qs=prof_ij["qs_layers"],
                vs_bedrock=760.0,
                density_bedrock=2.35,
                qs_bedrock=65.0,
                freq_range=(0.2, 5.0),
                n_freqs=120
            )
            pks, _ = find_peaks(hk_ij["amplification"], height=1.3, prominence=0.15, distance=5)
            f0_grid[i, j] = float(hk_ij["freqs"][pks[0]]) if len(pks) > 0 else float(hk_ij["f0"])
            amp_grid[i, j] = float(hk_ij["amplification"][pks[0]]) if len(pks) > 0 else float(hk_ij["peak_amp"])

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7.5))
    lon_m, lat_m = np.meshgrid(lons_cn, lats_cn)

    # (a) 共振主頻 f0 微區劃
    c1 = ax3a.contourf(lon_m, lat_m, f0_grid, levels=20, cmap="viridis_r")
    cs1 = ax3a.contour(lon_m, lat_m, f0_grid, levels=[0.8, 1.0, 1.2, 1.5, 2.0], colors="black", linewidths=0.8, linestyles="--")
    ax3a.clabel(cs1, inline=True, fontsize=8, fmt="%.1f Hz")
    cb1 = fig3.colorbar(c1, ax=ax3a, orientation="vertical", pad=0.02)
    cb1.set_label("共振基頻 Resonance Frequency f0 (Hz)", fontsize=10.5, fontweight="bold")

    # 標示嘉南地區強震站
    df_cn = df_res[df_res["basin"] == "Chianan"]
    ax3a.scatter(df_cn["lon"], df_cn["lat"], color="white", edgecolor="black", s=70, marker="^", label="TSMIP 強震測站", zorder=5)
    for _, r in df_cn.iterrows():
        ax3a.annotate(f"{r['station_code']}\n({r['f0_obs_hz']:.2f}Hz)", xy=(r["lon"], r["lat"]), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"))

    # 標注構造位置
    ax3a.text(120.25, 23.58, "北港基盤高區\n(淺基盤/高頻區)", fontsize=9.5, fontweight="bold", color="#FFE66D", bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7))
    ax3a.text(120.22, 23.22, "臺南前陸深槽\n(厚沉積/長週期低頻區)", fontsize=9.5, fontweight="bold", color="#FFE66D", bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7))

    ax3a.set_title("(a) 嘉南盆地強震長週期共振基頻 (f0) 空間微區劃", fontsize=12, fontweight="bold")
    ax3a.set_xlabel("經度 Longitude (°E)", fontsize=11, fontweight="bold")
    ax3a.set_ylabel("緯度 Latitude (°N)", fontsize=11, fontweight="bold")
    ax3a.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax3a.grid(True, linestyle=":", alpha=0.3)

    # (b) 共振峰值放大倍率 A_max 微區劃
    c2 = ax3b.contourf(lon_m, lat_m, amp_grid, levels=20, cmap="magma")
    cs2 = ax3b.contour(lon_m, lat_m, amp_grid, levels=[3.5, 4.0, 4.5, 5.0, 6.0], colors="white", linewidths=0.8, linestyles="--")
    ax3b.clabel(cs2, inline=True, fontsize=8, fmt="%.1f")
    cb2 = fig3.colorbar(c2, ax=ax3b, orientation="vertical", pad=0.02)
    cb2.set_label("共振峰值放大倍率 Peak Amplification", fontsize=10.5, fontweight="bold")

    ax3b.scatter(df_cn["lon"], df_cn["lat"], color="cyan", edgecolor="black", s=70, marker="^", label="TSMIP 強震測站", zorder=5)
    for _, r in df_cn.iterrows():
        ax3b.annotate(f"{r['station_code']}\n(Amp={r['peak_amp_obs']:.1f})", xy=(r["lon"], r["lat"]), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold", color="black",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax3b.set_title("(b) 嘉南盆地場址共振峰值放大倍率 (Peak Amplification) 空間分布", fontsize=12, fontweight="bold")
    ax3b.set_xlabel("經度 Longitude (°E)", fontsize=11, fontweight="bold")
    ax3b.set_ylabel("緯度 Latitude (°N)", fontsize=11, fontweight="bold")
    ax3b.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax3b.grid(True, linestyle=":", alpha=0.3)

    plt.suptitle("嘉南前陸盆地強震長週期共振基頻與場址放大倍率三維體素空間微區劃地圖\n3D Voxel Model Seismic Microzonation Map of Chianan Foreland Basin", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig3_path = reports_dir / "chianan_microzonation_map.png"
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"嘉南盆地空間微區劃地圖已儲存至: {fig3_path}")


if __name__ == "__main__":
    run_site_effects_validation()
