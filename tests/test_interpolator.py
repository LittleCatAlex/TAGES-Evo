import numpy as np
from pathlib import Path
from tages.features.interpolator import VelocityModel3D

def test_velocity_interpolator():
    csv_path = Path("data/interim/chiayi_velocity_3d.csv")
    assert csv_path.exists(), "需先有 3D 速度網格檔"
    
    vm = VelocityModel3D(csv_path)
    
    # 測試嘉義市中心 (約 23.48N, 120.45E) 於 100m 處
    res = vm.query(23.48, 120.45, 100.0)
    
    assert "vp" in res and "vs" in res and "vp_vs_ratio" in res and "dvs_dz" in res
    assert 2000 < res["vp"][0] < 5000, f"Vp={res['vp'][0]} 不在合理地殼範圍內"
    assert 1000 < res["vs"][0] < 3000, f"Vs={res['vs'][0]} 不在合理地殼範圍內"
    assert 1.4 < res["vp_vs_ratio"][0] < 2.5, f"Vp/Vs={res['vp_vs_ratio'][0]} 不合理"
    
def test_velocity_interpolator_vectorized():
    csv_path = Path("data/interim/chiayi_velocity_3d.csv")
    vm = VelocityModel3D(csv_path)
    
    lats = np.array([23.3, 23.4, 23.5])
    lons = np.array([120.2, 120.3, 120.4])
    depths = np.array([50.0, 200.0, 500.0])
    
    res = vm.query(lats, lons, depths)
    assert len(res["vp"]) == 3
    assert len(res["vs"]) == 3
    # 深度越深，波速通常越快
    assert res["vs"][2] > res["vs"][0]
