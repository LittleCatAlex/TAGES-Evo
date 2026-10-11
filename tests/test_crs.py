import math

from tages.geo.crs import convert


def test_roundtrip_wgs84_twd97():
    lon, lat = 120.45, 23.48  # 嘉義附近
    x, y = convert(lon, lat, "wgs84", "twd97")
    lon2, lat2 = convert(x, y, "twd97", "wgs84")
    assert math.isclose(lon, lon2, abs_tol=1e-8)
    assert math.isclose(lat, lat2, abs_tol=1e-8)


def test_twd97_value_in_range():
    x, y = convert(120.45, 23.48, "wgs84", "twd97")
    assert 150000 < x < 300000      # 嘉義約 E 200 km
    assert 2500000 < y < 2700000    # N 約 2597 km


def test_twd67_shift_is_small():
    # TWD67 與 TWD97 相差約 800 m 以內
    x67, y67 = 200000.0, 2597000.0
    x97, y97 = convert(x67, y67, "twd67", "twd97")
    assert abs(x97 - x67) < 1500 and abs(y97 - y67) < 1500
