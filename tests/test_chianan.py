"""西南臺灣 (嘉南平原與雲嘉南地區) 擴展範圍單元測試 (Unit Tests for Chianan Regional Extension)。"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

def test_chianan_velocity_grid():
    csv_path = Path("data/interim/chianan_velocity_3d.csv")
    assert csv_path.exists(), "需存在西南臺灣 3D 震波速度網格檔"

    df = pd.read_csv(csv_path)
    assert len(df) >= 1000, f"速度點數過少: {len(df)}"
    assert df["lat"].min() <= 23.00 and df["lat"].max() >= 23.80, "緯度需涵蓋 23.00 ~ 23.80"
    assert df["lon"].min() <= 120.05 and df["lon"].max() >= 120.70, "經度需涵蓋 120.05 ~ 120.70"
    assert df["depth_m"].max() >= 5000, "深度需涵蓋至深部基盤"
    assert (df["vs_ms"] > 1000).all() and (df["vs_ms"] < 4000).all(), "剪力波速 Vs 需在合理範圍內"

def test_chianan_voxels_file():
    npz_path = Path("data/processed/chianan_3d_voxels_1km.npz")
    assert npz_path.exists(), "需存在西南臺灣 3D 體素檔"

    data = np.load(npz_path)
    assert "lithology_3d" in data and "bedrock_depth_2d" in data
    assert "lats" in data and "lons" in data and "depths" in data

    lith = data["lithology_3d"]
    assert lith.shape == (46, 46, 101)
    assert lith.size == 213716

    # 檢查岩性類別是否在 1~10 規範內
    unique_lith = np.unique(lith)
    assert np.all((unique_lith >= 1) & (unique_lith <= 10))

    # 檢查基盤深度
    b2d = data["bedrock_depth_2d"]
    assert b2d.shape == (46, 46)
    assert b2d.min() >= 0.0 and b2d.max() <= 1000.0

def test_chianan_web_assets():
    html_path = Path("reports/chianan_3d_viewer.html")
    js_path = Path("reports/chianan_3d_data.js")
    assert html_path.exists() and html_path.stat().st_size > 5000
    assert js_path.exists() and js_path.stat().st_size > 500000

def test_chianan_ssr_summary():
    csv_path = Path("reports/chianan_ssr_summary.csv")
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) >= 6, f"需有至少 6 座代表性測站，目前為 {len(df)}"
    f0_col = "f0_model_hz" if "f0_model_hz" in df.columns else "f0_SSR_Hz"
    assert f0_col in df.columns
    assert (df[f0_col] > 0.1).all() and (df[f0_col] < 15.0).all()
