"""座標轉換工具：TWD67 / TWD97 / WGS84。

- TWD97 TM2 121°: EPSG:3826；TWD67 TM2 121°: EPSG:3828；WGS84: EPSG:4326
- 注意：TWD67→TWD97 在 pyproj 內使用公告之轉換參數，精度約公尺級，
  對 500 m 網格的 MVP 已足夠；若需更高精度，改用內政部官方六參數/網格。
- 本專案內部計算一律使用 TWD97 (EPSG:3826)。
"""
from __future__ import annotations

from pyproj import Transformer

_T = {
    ("wgs84", "twd97"): Transformer.from_crs(4326, 3826, always_xy=True),
    ("twd97", "wgs84"): Transformer.from_crs(3826, 4326, always_xy=True),
    ("twd67", "twd97"): Transformer.from_crs(3828, 3826, always_xy=True),
    ("twd97", "twd67"): Transformer.from_crs(3826, 3828, always_xy=True),
    ("twd67", "wgs84"): Transformer.from_crs(3828, 4326, always_xy=True),
    ("wgs84", "twd67"): Transformer.from_crs(4326, 3828, always_xy=True),
}


def convert(x, y, src: str, dst: str):
    """轉換座標。

    wgs84 的順序為 (lon, lat)；twd67/twd97 的順序為 (E, N)（即 x=E, y=N）。
    """
    src, dst = src.lower(), dst.lower()
    if src == dst:
        return x, y
    return _T[(src, dst)].transform(x, y)


def in_taiwan_wgs84(lon, lat) -> bool:
    """粗略檢查座標是否落在臺灣本島/離島合理範圍（排除 OCR 錯誤）。"""
    return 118.0 <= lon <= 122.5 and 21.5 <= lat <= 26.5
