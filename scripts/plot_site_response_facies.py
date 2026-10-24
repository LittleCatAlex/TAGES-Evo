"""沉積相物理特性對強震場址效應控制機制、反應譜分異與嘉南前陸盆地東西剖面剖析繪圖腳本。

本腳本產出圖 3-12（reports/site_response_spectra_facies.png）與量化指標檔（reports/site_response_metrics.json），
所有圖例與報告數字均由程式陣列實測計算（嚴禁硬編碼）：
(a) 不同沉積相（軟弱黏土、沖積中粗砂、山前粗礫石）之 Haskell–Thomson 頻率傳遞函數 |H(f)| 對比。
(b) 5% 阻尼比近似譜放大倍率 Sa(T) / PGA 及其建築敏感週期帶（峰值週期與放大係數由陣列動態計算）。
(c) 沉積/基盤波阻抗比 alpha 與共振放大倍率 A_max 之物理非線性控制曲線與 32 座測站對比。
(d) 嘉南前陸盆地由東向西（梅山 CHY006 -> 白河 CHY015 -> 新港 CHY039 -> 新營 CHY054 -> 下營 CHY017 -> 將軍 CHY016）共振週期長化與放大倍率演化剖析。
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.seismic.wave_propagator import compute_haskell_transfer_function

def generate_site_response_facies_plot():
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    df_truth = pd.read_csv(reports_dir / "tsmip_closed_loop_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # -------------------------------------------------------------
    # 子圖 (a): 不同沉積相一維 Haskell–Thomson 理論頻率放大譜 |H(f)|
    # -------------------------------------------------------------
    vs_bedrock = 760.0
    rho_bedrock = 2.35
    zb = rho_bedrock * vs_bedrock

    # 黏土相: H = 55m, Vs = 160 m/s, rho = 1.85, Qs = 18
    hk_clay = compute_haskell_transfer_function(
        thicknesses=[55.0], vs=[160.0], densities=[1.85], qs=[18.0],
        vs_bedrock=vs_bedrock, density_bedrock=rho_bedrock, qs_bedrock=65.0,
        freq_range=(0.1, 8.0), n_freqs=400
    )
    # 砂相: H = 45m, Vs = 220 m/s, rho = 1.90, Qs = 24
    hk_sand = compute_haskell_transfer_function(
        thicknesses=[45.0], vs=[220.0], densities=[1.90], qs=[24.0],
        vs_bedrock=vs_bedrock, density_bedrock=rho_bedrock, qs_bedrock=65.0,
        freq_range=(0.1, 8.0), n_freqs=400
    )
    # 礫石相: H = 35m, Vs = 360 m/s, rho = 2.10, Qs = 32
    hk_gravel = compute_haskell_transfer_function(
        thicknesses=[35.0], vs=[360.0], densities=[2.10], qs=[32.0],
        vs_bedrock=vs_bedrock, density_bedrock=rho_bedrock, qs_bedrock=65.0,
        freq_range=(0.1, 8.0), n_freqs=400
    )

    f0_c, amp_c = float(hk_clay['f0']), float(hk_clay['peak_amp'])
    f0_s, amp_s = float(hk_sand['f0']), float(hk_sand['peak_amp'])
    f0_g, amp_g = float(hk_gravel['f0']), float(hk_gravel['peak_amp'])

    alpha_c = float((1.85 * 160.0) / zb)
    alpha_s = float((1.90 * 220.0) / zb)
    alpha_g = float((2.10 * 360.0) / zb)

    ax1 = axes[0, 0]
    ax1.plot(hk_clay['freqs'], hk_clay['amplification'], color='#E63946', lw=2.6,
             label=f'深槽軟弱黏土相 (Clay: H=55m, Vs=160m/s, Amax={amp_c:.2f})')
    ax1.plot(hk_sand['freqs'], hk_sand['amplification'], color='#F4A261', lw=2.6,
             label=f'沖積砂質中層相 (Sand: H=45m, Vs=220m/s, Amax={amp_s:.2f})')
    ax1.plot(hk_gravel['freqs'], hk_gravel['amplification'], color='#2A9D8F', lw=2.6,
             label=f'山前粗礫石相 (Gravel: H=35m, Vs=360m/s, Amax={amp_g:.2f})')

    ax1.axvline(f0_c, color='#E63946', linestyle='--', alpha=0.75, label=f'黏土基頻 f0={f0_c:.2f} Hz (T0={1.0/f0_c:.2f} s)')
    ax1.axvline(f0_s, color='#F4A261', linestyle='--', alpha=0.75, label=f'砂層基頻 f0={f0_s:.2f} Hz (T0={1.0/f0_s:.2f} s)')
    ax1.axvline(f0_g, color='#2A9D8F', linestyle='--', alpha=0.75, label=f'礫石基頻 f0={f0_g:.2f} Hz (T0={1.0/f0_g:.2f} s)')

    ax1.set_title('(a) 不同沉積相一維 Haskell–Thomson 理論頻率放大譜 |H(f)|', fontsize=12, fontweight='bold')
    ax1.set_xlabel('頻率 Frequency (Hz)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('場址放大倍率 Amplification |H(f)|', fontsize=11, fontweight='bold')
    ax1.set_xlim(0.1, 8.0)
    ax1.set_ylim(0, 9.0)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', fontsize=8.5, frameon=True)

    # -------------------------------------------------------------
    # 子圖 (b): 加速度反應譜近似譜放大係數 Sa(T) / PGA 對比（全部由陣列動態計算）
    # -------------------------------------------------------------
    periods = np.logspace(-1.3, 0.6, 250)  # 0.05s ~ 4.0s

    def calc_rs_amplification(f_trans, amp_trans, f0_fundamental, T_eval, xi=0.05):
        f_eval = 1.0 / T_eval
        res = np.zeros(len(T_eval))
        for i, f_e in enumerate(f_eval):
            h_amp = float(np.interp(f_e, f_trans, amp_trans))
            omega_ratio = f_e / f0_fundamental
            d_factor = 1.0 / np.sqrt((1.0 - omega_ratio**2)**2 + (2.0 * xi * omega_ratio)**2)
            d_factor = np.clip(d_factor, 0.25, 2.5)
            res[i] = 1.0 + 0.62 * (h_amp - 1.0) * (d_factor / 2.5)
        return res

    sa_clay = calc_rs_amplification(hk_clay['freqs'], hk_clay['amplification'], f0_c, periods)
    sa_sand = calc_rs_amplification(hk_sand['freqs'], hk_sand['amplification'], f0_s, periods)
    sa_gravel = calc_rs_amplification(hk_gravel['freqs'], hk_gravel['amplification'], f0_g, periods)

    idx_c = int(np.argmax(sa_clay))
    idx_s = int(np.argmax(sa_sand))
    idx_g = int(np.argmax(sa_gravel))

    t_peak_c, sa_max_c = float(periods[idx_c]), float(sa_clay[idx_c])
    t_peak_s, sa_max_s = float(periods[idx_s]), float(sa_sand[idx_s])
    t_peak_g, sa_max_g = float(periods[idx_g]), float(sa_gravel[idx_g])

    ax2 = axes[0, 1]
    ax2.plot(periods, sa_clay, color='#E63946', lw=2.6,
             label=f'深槽軟弱黏土相 (峰值週期 T={t_peak_c:.2f}s, Sa/PGA={sa_max_c:.2f})')
    ax2.plot(periods, sa_sand, color='#F4A261', lw=2.6,
             label=f'沖積砂質中層相 (峰值週期 T={t_peak_s:.2f}s, Sa/PGA={sa_max_s:.2f})')
    ax2.plot(periods, sa_gravel, color='#2A9D8F', lw=2.6,
             label=f'山前粗礫石相 (峰值週期 T={t_peak_g:.2f}s, Sa/PGA={sa_max_g:.2f})')

    ax2.axvspan(0.1, 0.4, color='#2A9D8F', alpha=0.12, label='低矮透天/剛性結構敏感帶 (T=0.1~0.4s)')
    ax2.axvspan(0.5, 1.0, color='#F4A261', alpha=0.12, label='中低樓層RC建築敏感帶 (T=0.5~1.0s)')
    ax2.axvspan(1.1, 2.2, color='#E63946', alpha=0.12, label='高層建築/長跨結構敏感帶 (T=1.1~2.2s)')

    ax2.set_xscale('log')
    ax2.set_title('(b) 5% 阻尼比近似譜放大係數 Sa(T) / PGA（傳遞譜耦合 SDOF 轉換）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('結構自振週期 Period T (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('近似譜放大係數 Sa(T) / PGA', fontsize=11, fontweight='bold')
    ax2.set_xlim(0.05, 4.0)
    ax2.set_ylim(0.8, 6.2)
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=8.5, frameon=True)

    # -------------------------------------------------------------
    # 子圖 (c): 波阻抗比 alpha 與共振放大倍率 A_max 之物理非線性控制
    # -------------------------------------------------------------
    alpha_obs = (df_truth['vs30_obs_ms'] * 1.90) / zb
    amp_obs = df_truth['peak_amp_obs'].values
    alpha_voxel = (df_truth['vs30_voxel_ms'] * 1.90) / zb
    amp_voxel = df_truth['peak_amp_model'].values

    alpha_theory = np.linspace(0.12, 0.85, 200)
    amp_damped_20 = 1.0 / (alpha_theory + np.pi / (4.0 * 20.0))
    amp_damped_35 = 1.0 / (alpha_theory + np.pi / (4.0 * 35.0))

    ax3 = axes[1, 0]
    ax3.plot(alpha_theory, amp_damped_20, color='#457B9D', linestyle='-', lw=2.2, label='理論黏彈阻尼放大曲線 (Qs=20)')
    ax3.plot(alpha_theory, amp_damped_35, color='#1D3557', linestyle='--', lw=2.2, label='理論黏彈阻尼放大曲線 (Qs=35)')
    ax3.scatter(alpha_voxel, amp_voxel, color='#E63946', s=85, edgecolors='black', label=f'本研究 3D 體素正演峰值 (N={len(df_truth)})', zorder=5)
    ax3.scatter(alpha_obs, amp_obs, color='#111111', marker='x', s=65, label=f'EGDT 經驗基準峰值 (N={len(df_truth)})', zorder=4)

    ax3.set_title('(c) 沉積/基盤波阻抗比 alpha 與共振放大倍率 A_max 之物理非線性控制', fontsize=12, fontweight='bold')
    ax3.set_xlabel('波阻抗比 alpha = (rho1 * Vs1) / (rho_b * Vs_b)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('共振放大峰值 A_max', fontsize=11, fontweight='bold')
    ax3.set_xlim(0.12, 0.85)
    ax3.set_ylim(1.5, 9.0)
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='upper right', fontsize=9, frameon=True)

    # -------------------------------------------------------------
    # 子圖 (d): 嘉南前陸盆地由東向西剖面（山前階地至臺南深槽）共振週期長化剖析
    # -------------------------------------------------------------
    tai_stations = ['CHY029', 'CHY006', 'CHY015', 'CHY039', 'CHY054', 'CHY016']
    tai_names = [
        '古坑 CHY029\n(山前麓山)',
        '梅山 CHY006\n(山前階地)',
        '白河 CHY015\n(平原東側)',
        '新港 CHY039\n(北港翼側)',
        '新營 CHY054\n(深槽北側)',
        '將軍 CHY016\n(濱海深槽)'
    ]
    tai_df = df_truth.set_index('station_code').loc[tai_stations].reset_index()

    x_idx = np.arange(len(tai_stations))
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    t0_voxel = 1.0 / tai_df['f0_voxel_hz'].values
    t0_obs = 1.0 / tai_df['f0_obs_hz'].values

    ax4.bar(x_idx - 0.18, t0_voxel, width=0.35, color='#457B9D', alpha=0.85, label='3D 體素正演共振週期 T0_voxel (s)', edgecolor='black')
    ax4.bar(x_idx + 0.18, t0_obs, width=0.35, color='#E63946', alpha=0.85, label='基準共振週期 T0_obs (s)', edgecolor='black')

    ax4_twin.plot(x_idx, tai_df['peak_amp_obs'], color='#111111', marker='o', lw=2.4, markersize=8, label='基準共振放大倍率 Peak Amp')
    ax4_twin.plot(x_idx, tai_df['peak_amp_model'], color='#E76F51', marker='s', linestyle='--', lw=2.2, markersize=7, label='體素正演放大倍率 Peak Amp')

    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(tai_names, fontsize=9.5, fontweight='bold')
    ax4.set_ylabel('共振卓越週期 T0 = 1 / f0 (s)', fontsize=11, fontweight='bold', color='#1D3557')
    ax4_twin.set_ylabel('場址放大倍率 Peak Amplification', fontsize=11, fontweight='bold', color='#E63946')
    ax4.set_title('(d) 嘉南前陸盆地由東向西強震共振週期長化與放大倍率演化剖析', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 2.0)
    ax4_twin.set_ylim(1.5, 9.0)
    ax4.grid(True, linestyle='--', alpha=0.4)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8.5, frameon=True)

    plt.suptitle('沉積相物理特性對強震場址效應控制機制、反應譜分異與盆地剖面剖析\nPhysics Control of Sedimentary Facies on Strong-Motion Site Effects and Response Spectra', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    out_path = reports_dir / "site_response_spectra_facies.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 輸出實測計算之量化指標至 JSON，供說明書引用（0% 硬編碼）
    transect_records = []
    for idx_t, r_t in tai_df.iterrows():
        transect_records.append({
            "station_code": r_t["station_code"],
            "station_name": r_t["station_name"],
            "vs30_obs_ms": float(r_t["vs30_obs_ms"]),
            "z_bedrock_ac_m": float(r_t["z_bedrock_ac_m"]),
            "f0_voxel_hz": float(r_t["f0_voxel_hz"]),
            "t0_voxel_s": round(float(1.0 / r_t["f0_voxel_hz"]), 2),
            "f0_obs_hz": float(r_t["f0_obs_hz"]),
            "t0_obs_s": round(float(1.0 / r_t["f0_obs_hz"]), 2),
            "peak_amp_model": float(r_t["peak_amp_model"]),
            "peak_amp_obs": float(r_t["peak_amp_obs"])
        })

    metrics_json = {
        "facies_theoretical_models": {
            "clay_soft": {
                "h_m": 55.0,
                "vs_ms": 160.0,
                "rho_gcc": 1.85,
                "qs": 18.0,
                "impedance_ratio_alpha": round(alpha_c, 3),
                "f0_hz": round(f0_c, 2),
                "t0_s": round(1.0 / f0_c, 2),
                "peak_amp_h": round(amp_c, 2),
                "rs_peak_period_s": round(t_peak_c, 2),
                "rs_peak_sa_pga": round(sa_max_c, 2)
            },
            "sand_medium": {
                "h_m": 45.0,
                "vs_ms": 220.0,
                "rho_gcc": 1.90,
                "qs": 24.0,
                "impedance_ratio_alpha": round(alpha_s, 3),
                "f0_hz": round(f0_s, 2),
                "t0_s": round(1.0 / f0_s, 2),
                "peak_amp_h": round(amp_s, 2),
                "rs_peak_period_s": round(t_peak_s, 2),
                "rs_peak_sa_pga": round(sa_max_s, 2)
            },
            "gravel_stiff": {
                "h_m": 35.0,
                "vs_ms": 360.0,
                "rho_gcc": 2.10,
                "qs": 32.0,
                "impedance_ratio_alpha": round(alpha_g, 3),
                "f0_hz": round(f0_g, 2),
                "t0_s": round(1.0 / f0_g, 2),
                "peak_amp_h": round(amp_g, 2),
                "rs_peak_period_s": round(t_peak_g, 2),
                "rs_peak_sa_pga": round(sa_max_g, 2)
            }
        },
        "station_impedance_stats": {
            "alpha_voxel_min": round(float(alpha_voxel.min()), 3),
            "alpha_voxel_max": round(float(alpha_voxel.max()), 3),
            "alpha_obs_min": round(float(alpha_obs.min()), 3),
            "alpha_obs_max": round(float(alpha_obs.max()), 3),
            "amp_voxel_min": round(float(amp_voxel.min()), 2),
            "amp_voxel_max": round(float(amp_voxel.max()), 2),
            "amp_obs_min": round(float(amp_obs.min()), 2),
            "amp_obs_max": round(float(amp_obs.max()), 2)
        },
        "chianan_ew_transect": transect_records
    }

    metrics_path = reports_dir / "site_response_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(f"圖表已成功輸出至: {out_path}")
    print(f"反應譜與沉積相量化指標已輸出至: {metrics_path}")

if __name__ == "__main__":
    generate_site_response_facies_plot()
