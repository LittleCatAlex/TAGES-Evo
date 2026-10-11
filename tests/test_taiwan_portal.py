"""全臺灣 2D GIS 門戶與多盆地高密度 3D 體素管線單元測試 (Unit Tests for Taiwan Portal & Multi-Basin Dense 3D Pipeline)。"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

def test_taiwan_boreholes():
    bh_path = Path("data/interim/taiwan_boreholes.csv")
    assert bh_path.exists(), "需存在全台水文觀測井資料庫"

    df = pd.read_csv(bh_path)
    assert len(df) == 423, f"全島水文觀測井應為 423 口，目前為 {len(df)}"
    assert (df["lat"] >= 21.8).all() and (df["lat"] <= 25.5).all(), "緯度需在臺灣本島與離島範圍內"
    assert (df["lon"] >= 119.5).all() and (df["lon"] <= 122.2).all(), "經度需在臺灣本島與離島範圍內"
    assert (df["well_depth_m"] > 0).all(), "井深需大於 0 公尺"

def test_taiwan_velocity_grid():
    vel_path = Path("data/interim/taiwan_velocity_3d.csv")
    assert vel_path.exists(), "需存在全島 3D 震波速度網格"

    df = pd.read_csv(vel_path)
    assert len(df) >= 2500, f"速度點數過少: {len(df)}"
    assert df["lat"].min() <= 22.5 and df["lat"].max() >= 25.0, "需涵蓋南臺灣至北臺灣"
    assert (df["vs_ms"] >= 1000).all() and (df["vs_ms"] <= 4500).all()

def test_taiwan_basins_dense_npz():
    npz_path = Path("data/processed/taiwan_basins_dense.npz")
    assert npz_path.exists(), "需存在六大盆地高密度 3D 體素檔案"

    data = np.load(npz_path)
    required_basins = ["chianan", "zhuoshui", "taipei", "taichung", "pingtung", "yilan"]
    for b in required_basins:
        assert f"{b}_lith" in data, f"缺少盆地 {b} 之 3D 岩性數據"
        assert f"{b}_vs" in data, f"缺少盆地 {b} 之 3D 波速數據"
        assert f"{b}_bedrock" in data, f"缺少盆地 {b} 之基盤數據"
        lith = data[f"{b}_lith"]
        assert lith.ndim == 3 and lith.size > 20000, f"盆地 {b} 體素點數過少: {lith.size}"
        # 驗證岩性分類在 1~10 規範內
        unq = np.unique(lith)
        assert np.all((unq >= 1) & (unq <= 10))

def test_taiwan_portal_assets():
    html_path = Path("reports/taiwan_3d_portal.html")
    js_path = Path("reports/taiwan_portal_data.js")
    assert html_path.exists() and html_path.stat().st_size > 10000
    assert js_path.exists() and js_path.stat().st_size > 5000000

    html_content = html_path.read_text(encoding="utf-8")
    assert "cartocdn" not in html_content.lower(), "不可使用需付費或強制要求 API key 的 Carto 圖資"
    assert "arcgisonline" in html_content, "需支援免 API Key 之 Esri 圖資"
    assert "openstreetmap" in html_content, "需支援免 API Key 之 OpenStreetMap 圖資"
    assert "control.layers" in html_content, "需具備圖層切換控制元件"
