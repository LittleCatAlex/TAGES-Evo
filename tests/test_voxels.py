from pathlib import Path
import numpy as np
import pytest
from tages.voxels.volume import VoxelGridConfig, VoxelVolumeGenerator, estimate_elevation

def test_estimate_elevation():
    lons = np.array([120.10, 120.35, 120.60])
    lats = np.array([23.40, 23.40, 23.40])
    elev = estimate_elevation(lons, lats)
    assert len(elev) == 3
    assert elev[0] < elev[1] < elev[2]
    assert 2.0 <= elev[0] <= 10.0
    assert elev[2] > 50.0

def test_voxel_volume_small_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    vel_csv = root / "data" / "interim" / "chiayi_velocity_3d.csv"
    model_pkl = root / "models" / "tages_rf_augmented.pkl"

    if not vel_csv.exists() or not model_pkl.exists():
        pytest.skip("Velocity CSV or model PKL not available for test.")

    cfg = VoxelGridConfig(
        lon_min=120.20, lon_max=120.30, n_lon=3,
        lat_min=23.40, lat_max=23.45, n_lat=2,
        depth_min=0.0, depth_max=100.0, n_depth=5
    )

    gen = VoxelVolumeGenerator(vel_csv, model_pkl, config=cfg)
    data = gen.generate()

    assert data["lithology_3d"].shape == (2, 3, 5)
    assert data["confidence_3d"].shape == (2, 3, 5)
    assert data["vs_3d"].shape == (2, 3, 5)
    assert data["bedrock_depth_2d"].shape == (2, 3)
    assert not np.isnan(data["lithology_3d"]).any()

    # 測試儲存 NPZ
    out_npz = tmp_path / "test_vol.npz"
    gen.save_volume(data, out_npz)
    assert out_npz.exists()
    assert out_npz.stat().st_size > 0
