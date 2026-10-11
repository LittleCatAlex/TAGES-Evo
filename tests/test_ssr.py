import numpy as np
from tages.seismic.ssr import compute_spectrum, compute_ssr, konno_ohmachi_smooth

def test_spectrum_and_smoothing():
    dt = 0.01  # 100 Hz sampling
    t = np.linspace(0, 10, 1000, endpoint=False)
    # 2 Hz signal
    sig = np.sin(2 * np.pi * 2.0 * t)
    freqs, spec = compute_spectrum(sig, dt, smooth_b=40.0)
    
    assert len(freqs) == len(spec)
    assert freqs[0] == 0.0
    # Peak should be near 2 Hz
    peak_f = freqs[np.argmax(spec)]
    assert abs(peak_f - 2.0) < 0.2

def test_ssr_amplification():
    dt = 0.01
    t = np.linspace(0, 10, 1000, endpoint=False)
    # Target station has 3x amplification at 1.5 Hz
    sig_ref = np.sin(2 * np.pi * 1.5 * t)
    sig_target = 3.0 * np.sin(2 * np.pi * 1.5 * t)
    
    res = compute_ssr(sig_target, sig_ref, dt)
    assert res["valid"] is True
    # Amplification near 1.5 Hz should be around 3.0
    assert abs(res["peak_amp"] - 3.0) < 0.5
