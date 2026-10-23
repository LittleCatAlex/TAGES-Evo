import numpy as np
import pytest
from tages.seismic.wave_propagator import (
    compute_haskell_transfer_function,
    compute_quarter_wavelength_f0,
    build_voxel_1d_stratified_profile,
    FACIES_ROCK_PHYSICS
)

def test_haskell_transfer_function_single_layer():
    # Single layer of thickness 30m, Vs = 200 m/s over bedrock Vs = 760 m/s
    # Theoretical quarter-wavelength fundamental frequency f0 = Vs / (4H) = 200 / (4 * 30) = 1.67 Hz
    thicks = np.array([30.0])
    vs = np.array([200.0])
    densities = np.array([1.90])
    qs = np.array([25.0])

    res = compute_haskell_transfer_function(
        thicknesses=thicks,
        vs=vs,
        densities=densities,
        qs=qs,
        vs_bedrock=760.0,
        density_bedrock=2.35,
        freq_range=(0.2, 5.0),
        n_freqs=300
    )

    assert "f0" in res
    assert "amplification" in res
    assert abs(res["f0"] - 1.67) < 0.15, f"Expected f0 ~ 1.67 Hz, got {res['f0']}"
    assert res["peak_amp"] > 2.0, "Fundamental resonance should have amplification > 2.0"

def test_quarter_wavelength_formula():
    f0 = compute_quarter_wavelength_f0(vs=200.0, h=50.0)
    assert abs(f0 - 1.0) < 1e-4

def test_build_voxel_1d_stratified_profile():
    depths = np.linspace(0, 300, 31)
    lith_col = np.full(31, 2)  # All clay
    lith_col[20:] = 8         # Mudstone bedrock

    prof = build_voxel_1d_stratified_profile(
        depths=depths,
        lith_col=lith_col,
        z_eng=60.0,
        z_ac=250.0,
        basin_key="chianan"
    )

    assert prof["h_eff_voxel_m"] > 20.0
    assert len(prof["thicks"]) == len(prof["vs_layers"])
    assert prof["vs30_voxel_ms"] > 100.0
    assert len(prof["vs_layers"]) >= 14
