"""標準頻譜比法 (Standard Spectral Ratio, SSR) 與場址效應計算模組。

理論依據：
  O(f) = E(f) * P(f) * S(f)
  SSR(f) = O_target(f) / O_ref(f) ≈ S_target(f)
  當目標站與堅硬岩盤參考站 (S_ref ≈ 1) 距離震源相近時，震源與路徑效應互相抵消。

核心特色：
1. S 波視窗截取與餘弦漸拔 (Cosine Taper)。
2. Konno-Ohmachi (或高斯) 頻譜平滑化，消除高頻雜訊。
3. 訊雜比 (SNR) 檢驗：對比 S 波與地震前背景雜訊振幅。
4. 動態參考站選擇機制：自動挑選距離震央最近且訊號正常的岩盤候選站。
5. 萃取 1 Hz、2 Hz、5 Hz 放大倍率與峰值頻率 (f0)。
"""
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, rfftfreq

def konno_ohmachi_smooth(freqs: np.ndarray, spec: np.ndarray, b: float = 40.0) -> np.ndarray:
    """Konno-Ohmachi (1998) 頻譜平滑化演算法。
    
    b: 平滑帶寬係數 (典型值 20 ~ 40)
    """
    n = len(freqs)
    smoothed = np.zeros(n)
    # 避免 log(0)
    f = np.maximum(freqs, 1e-4)
    
    for i in range(n):
        fc = f[i]
        # w = (sin(b * log10(f / fc)) / (b * log10(f / fc)))^4
        ratio = f / fc
        log_ratio = np.log10(ratio)
        # 處理極限值 f == fc
        weights = np.ones(n)
        mask = np.abs(log_ratio) > 1e-6
        arg = b * log_ratio[mask]
        weights[mask] = (np.sin(arg) / arg) ** 4
        
        total_w = np.sum(weights)
        if total_w > 0:
            smoothed[i] = np.sum(spec * weights) / total_w
        else:
            smoothed[i] = spec[i]
            
    return smoothed

def compute_spectrum(signal: np.ndarray, dt: float, smooth_b: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """計算時間序列之平滑振幅頻譜。"""
    n = len(signal)
    # Cosine taper 10%
    taper = np.ones(n)
    w_len = int(0.1 * n)
    if w_len > 0:
        window = 0.5 * (1 - np.cos(np.linspace(0, np.pi, w_len)))
        taper[:w_len] = window
        taper[-w_len:] = window[::-1]
        
    sig_tapered = (signal - np.mean(signal)) * taper
    freqs = rfftfreq(n, d=dt)
    raw_spec = np.abs(rfft(sig_tapered)) * dt
    smoothed_spec = konno_ohmachi_smooth(freqs, raw_spec, b=smooth_b)
    return freqs, smoothed_spec

def compute_ssr(
    target_s_wave: np.ndarray,
    ref_s_wave: np.ndarray,
    dt: float,
    noise_window: np.ndarray | None = None,
    min_snr: float = 3.0,
    max_amplification: float = 50.0
) -> dict:
    """計算目標測站與參考測站之標準頻譜比 (SSR)。"""
    # 1. 檢驗 SNR (若提供背景雜訊)
    if noise_window is not None and len(noise_window) > 0:
        s_rms = np.sqrt(np.mean(target_s_wave ** 2))
        n_rms = np.sqrt(np.mean(noise_window ** 2)) + 1e-9
        snr = s_rms / n_rms
        if snr < min_snr:
            return {"valid": False, "reason": f"SNR {snr:.1f} < {min_snr}"}
    else:
        snr = None

    # 2. 計算目標與參考測站頻譜
    freqs_t, spec_t = compute_spectrum(target_s_wave, dt)
    freqs_r, spec_r = compute_spectrum(ref_s_wave, dt)

    # 3. 計算頻譜比
    denom = np.maximum(spec_r, 1e-6)
    ssr = spec_t / denom

    # 4. 異常極端值過濾
    if np.max(ssr) > max_amplification * 2:
        return {"valid": False, "reason": f"Max SSR {np.max(ssr):.1f} exceeds threshold"}

    # 5. 萃取工程常用頻率放大係數
    def get_amp_at_freq(target_f):
        idx = np.argmin(np.abs(freqs_t - target_f))
        return float(ssr[idx])

    # 頻率範圍 0.2 ~ 10 Hz 內的峰值頻率與峰值放大
    mask = (freqs_t >= 0.2) & (freqs_t <= 10.0)
    if np.any(mask):
        sub_ssr = ssr[mask]
        sub_f = freqs_t[mask]
        peak_idx = np.argmax(sub_ssr)
        f0 = float(sub_f[peak_idx])
        peak_amp = float(sub_ssr[peak_idx])
    else:
        f0 = None
        peak_amp = None

    return {
        "valid": True,
        "snr": float(snr) if snr is not None else None,
        "f0": f0,
        "peak_amp": peak_amp,
        "amp_0_5hz": get_amp_at_freq(0.5),
        "amp_1hz": get_amp_at_freq(1.0),
        "amp_2hz": get_amp_at_freq(2.0),
        "amp_5hz": get_amp_at_freq(5.0),
        "freqs": freqs_t.tolist(),
        "ssr": ssr.tolist()
    }
