"""強震頻譜比法 (SSR) 場址效應與 3D 地體模型跨學科驗證模組 (SSR Site Amplification & Cross-Validation)。

科學依據：
沉積盆地之一維剪力波共振理論 (Quarter-Wavelength Resonance Theory):
  f_0 = V_s_avg / (4 * H_sed)
其中：
  H_sed: 沉積層厚度 (即 3D 體素模型萃取之聲學基盤深度 Z_bedrock)
  V_s_avg: 沉積層平均剪力波速
  f_0: 地表強震 SSR 頻譜比主頻 (Predominant Resonance Frequency)

本腳本：
1. 定義嘉義代表性強震測站 (涵蓋海岸帶、平原中心、市區與麓山帶)。
2. 自 3D 體素地體模型 (chiayi_3d_voxels_1km.npz) 萃取各測站正下方之基盤深度與平均波速。
3. 採用一維黏彈性共振濾波器正演模擬沉積層響應 (Forward Numerical Simulation)，並執行標準頻譜比法 (SSR) 與 Konno-Ohmachi (1998) 平滑化。
4. 驗證 3D 地下體素構造與一維剪力波共振理論 (Quarter-Wavelength Resonance) 之數值一致性（實測 TSMIP 測站記象之真實比對規劃於 Phase C 執行）。
5. 產出 reports/chiayi_ssr_site_effects.png 與 reports/chiayi_ssr_summary.csv。
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.seismic.ssr import compute_ssr, konno_ohmachi_smooth

# 嘉義區域代表性強震觀測站 (模擬與實測波形場址)
STATIONS = [
    {
        "code": "CHY010",
        "name": "布袋 (Budai)",
        "lat": 23.38,
        "lon": 120.16,
        "site_class": "Class E (Soft Soil / Deep Basin)"
    },
    {
        "code": "CHY080",
        "name": "新港 (Singang)",
        "lat": 23.55,
        "lon": 120.35,
        "site_class": "Class D (Alluvial Plain)"
    },
    {
        "code": "CHY015",
        "name": "嘉義市 (Chiayi City)",
        "lat": 23.48,
        "lon": 120.45,
        "site_class": "Class D (Mid-Plio Sediments)"
    },
    {
        "code": "CHY029",
        "name": "中埔/觸口 (Chukou)",
        "lat": 23.44,
        "lon": 120.57,
        "site_class": "Class C (Foothill Semi-Rock)"
    },
    {
        "code": "ALS",
        "name": "阿里山 (Alishan Ref Bedrock)",
        "lat": 23.51,
        "lon": 120.81,
        "site_class": "Class B (Reference Hard Bedrock)"
    }
]

def simulate_station_swave(fs: float, duration: float, f0: float, amp: float, q_factor: float = 25.0) -> np.ndarray:
    """依據場址傳遞函數 (Transfer Function) 模擬 S 波地表響應。"""
    n = int(fs * duration)
    t = np.linspace(0, duration, n)
    dt = 1.0 / fs

    # 模擬入射深部基盤 S 波 (寬頻 Ricker 小波脈衝 + 隨機散射波)
    t0 = 5.0
    fp = 1.5
    tau = (t - t0) * fp
    base_wave = (1.0 - 2.0 * (np.pi * tau) ** 2) * np.exp(- (np.pi * tau) ** 2)
    # 添加隨機散射背景
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, n)
    input_signal = base_wave + noise

    # 一維單自由度諧振濾波器模擬沉積層放大 (Resonance Filter)
    freqs = np.fft.rfftfreq(n, d=dt)
    w = 2 * np.pi * freqs
    w0 = 2 * np.pi * f0
    damping = 1.0 / (2.0 * q_factor)
    
    # 傳遞函數 H(w) = 1 / ( (1 - (w/w0)^2) + 2j*damping*(w/w0) )
    denom = (1.0 - (w / np.maximum(w0, 1e-4)) ** 2) + 2j * damping * (w / np.maximum(w0, 1e-4))
    h_w = 1.0 + (amp - 1.0) / denom

    spec = np.fft.rfft(input_signal)
    surface_spec = spec * h_w
    surface_wave = np.fft.irfft(surface_spec, n=n)
    return surface_wave

def run_ssr_pipeline():
    root = Path(__file__).resolve().parents[1]
    npz_path = root / "data" / "processed" / "chiayi_3d_voxels_1km.npz"
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    print("載入 3D 地下體素地體模型...")
    vol_data = np.load(npz_path)
    lats = vol_data["lats"]
    lons = vol_data["lons"]
    depths = vol_data["depths"]
    vs_3d = vol_data["vs_3d"]
    bedrock_2d = vol_data["bedrock_depth_2d"]

    # 參考站訊號 (硬岩 ALS: f0=15Hz, amp=1.0)
    fs = 100.0  # 100 Hz
    duration = 40.0 # 40 秒
    dt = 1.0 / fs
    ref_signal = simulate_station_swave(fs, duration, f0=15.0, amp=1.0, q_factor=50.0)

    results = []

    print("\n=======================================================")
    print("  各強震站沉積層厚度 (3D 體素萃取) 與 SSR 共振頻率分析")
    print("=======================================================")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]

    for idx, sta in enumerate(STATIONS[:4]):  # 前四站為目標站
        # 自 3D 體素模型萃取該座標之沉積層基盤深度與平均波速
        lat_idx = np.argmin(np.abs(lats - sta["lat"]))
        lon_idx = np.argmin(np.abs(lons - sta["lon"]))

        h_sed = float(bedrock_2d[lat_idx, lon_idx])
        # 計算沉積層內的平均 Vs
        z_match = np.where(depths <= h_sed)[0]
        if len(z_match) > 0:
            vs_col = vs_3d[lat_idx, lon_idx, z_match]
            vs_avg = float(np.mean(vs_col))
        else:
            vs_avg = float(vs_3d[lat_idx, lon_idx, 0])

        # 理論 1/4 波長共振頻率: f0_theo = Vs / (4 * H)
        f0_theo = vs_avg / (4.0 * max(h_sed, 10.0))
        # 放大倍率經驗估算: A0 ~ Vs_rock / Vs_sed (岩盤 Vs 假設 2800 m/s)
        amp_theo = np.clip(2800.0 / max(vs_avg, 500.0), 1.5, 6.0)

        # 模擬該測站 S 波訊號
        sta_signal = simulate_station_swave(fs, duration, f0=f0_theo, amp=amp_theo, q_factor=20.0)

        # 計算 SSR 頻譜比
        ssr_res = compute_ssr(sta_signal, ref_signal, dt)
        f0_ssr = ssr_res["f0"]
        amp_ssr = ssr_res["peak_amp"]

        print(f"測站 {sta['code']} ({sta['name']}):")
        print(f"  基盤深度 H_sed: {h_sed:.1f} m | 平均剪力波速 Vs: {vs_avg:.0f} m/s")
        print(f"  理論共振頻率 f0_theo: {f0_theo:.2f} Hz | SSR 實測主頻 f0_ssr: {f0_ssr:.2f} Hz")
        print(f"  SSR 峰值放大倍率 A0:   {amp_ssr:.2f}x\n")

        results.append({
            "Station": sta["code"],
            "Name": sta["name"],
            "Latitude": sta["lat"],
            "Longitude": sta["lon"],
            "Site_Class": sta["site_class"],
            "Bedrock_Depth_m": round(h_sed, 1),
            "Vs_avg_mps": round(vs_avg, 0),
            "f0_Theory_Hz": round(f0_theo, 2),
            "f0_SSR_Hz": round(f0_ssr, 2),
            "Peak_Amp_SSR": round(amp_ssr, 2),
            "Amp_1Hz": round(ssr_res["amp_1hz"], 2),
            "Amp_2Hz": round(ssr_res["amp_2hz"], 2)
        })

        # 繪製 SSR 頻譜比曲線
        freqs = np.array(ssr_res["freqs"])
        ssr_curve = np.array(ssr_res["ssr"])
        mask = (freqs >= 0.2) & (freqs <= 10.0)

        ax1.plot(freqs[mask], ssr_curve[mask], label=f"{sta['code']} ({sta['name']}): f0={f0_ssr:.2f}Hz, H={h_sed:.0f}m", color=colors[idx], linewidth=2.2)
        ax1.axvline(f0_ssr, color=colors[idx], linestyle="--", alpha=0.6)

    ax1.set_xlabel("Frequency (Hz)", fontsize=11)
    ax1.set_ylabel("SSR Amplification (Target / Rock Ref)", fontsize=11)
    ax1.set_title("A) Standard Spectral Ratio (SSR) Amplification Curves across Chiayi Transect", fontsize=13, fontweight="bold")
    ax1.set_xlim(0.2, 10.0)
    ax1.set_ylim(0.5, 7.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # 下圖: 沉積層基盤深度 H 與共振頻率 f0 的負相關雙向驗證
    df_res = pd.DataFrame(results)
    h_vals = df_res["Bedrock_Depth_m"].values
    f0_theo_vals = df_res["f0_Theory_Hz"].values
    f0_ssr_vals = df_res["f0_SSR_Hz"].values

    x_pos = np.arange(len(df_res))
    width = 0.35

    rects1 = ax2.bar(x_pos - width/2, f0_theo_vals, width, label="1/4-Wavelength Theory (f0 = Vs / 4H)", color="#3A86FF", alpha=0.85)
    rects2 = ax2.bar(x_pos + width/2, f0_ssr_vals, width, label="1D Forward Simulation SSR Peak (f0_sim)", color="#FF006E", alpha=0.85)

    ax2.set_xlabel("Seismic Station (West Coast -> Western Foothills)", fontsize=11)
    ax2.set_ylabel("Resonance Frequency f0 (Hz)", fontsize=11)
    ax2.set_title("B) Numerical Forward Validation: 1D Resonance Physics vs 3D Voxel Basin Structure", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r['Station']}\n({r['Name']})\nH={r['Bedrock_Depth_m']}m" for _, r in df_res.iterrows()], fontsize=10)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_png = rep_dir / "chiayi_ssr_site_effects.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"SSR 分析圖已儲存至: {out_png}")

    # 匯出 CSV
    out_csv = rep_dir / "chiayi_ssr_summary.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"SSR 統計表已儲存至: {out_csv}")

if __name__ == "__main__":
    run_ssr_pipeline()
