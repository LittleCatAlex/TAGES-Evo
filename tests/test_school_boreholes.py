"""學校工程鑽孔資料解析與特徵整合單元測試 (Unit Tests for School Boreholes Integration)"""
from pathlib import Path
import pandas as pd
import pytest

def test_school_boreholes_csv_exists_and_valid():
    root = Path(__file__).resolve().parents[1]
    bh_path = root / "data" / "interim" / "school_boreholes.csv"
    assert bh_path.exists(), f"找不到 {bh_path}"
    df_bh = pd.read_csv(bh_path)
    assert len(df_bh) >= 6000, f"學校工程鑽孔數應 >= 6,000，實得 {len(df_bh)}"
    assert "station" in df_bh.columns
    assert "lon" in df_bh.columns and "lat" in df_bh.columns
    # 經緯度範圍檢查 (臺灣本島)
    assert df_bh["lon"].between(120.0, 122.5).all(), "經度超出臺灣範圍"
    assert df_bh["lat"].between(21.8, 25.5).all(), "緯度超出臺灣範圍"


def test_school_strata_csv_validity():
    root = Path(__file__).resolve().parents[1]
    st_path = root / "data" / "interim" / "school_strata.csv"
    assert st_path.exists(), f"找不到 {st_path}"
    df_st = pd.read_csv(st_path)
    assert len(df_st) >= 380, f"學校地層層數過少: {len(df_st)}"
    assert (df_st["bottom_m"] >= df_st["top_m"]).all(), "存在底深小於頂深的地層"
    # 類別檢查 (L2 應介於 0 到 10，0 為未定義/其他地層)
    assert df_st["l2"].between(0, 10).all(), "存在無效 L2 標籤"


def test_combined_dataset_validity():
    root = Path(__file__).resolve().parents[1]
    comb_path = root / "data" / "processed" / "dataset_combined.csv"
    assert comb_path.exists(), f"找不到 {comb_path}"
    df_comb = pd.read_csv(comb_path)
    assert len(df_comb) > 118000, f"融合資料集樣本數過少: {len(df_comb)}"
    assert "station_id" in df_comb.columns
    assert "vp_ms" in df_comb.columns
    assert "vs_ms" in df_comb.columns
    assert "depth_m" in df_comb.columns
    # 關鍵特徵不得有 NaN
    assert df_comb["vp_ms"].notna().all()
    assert df_comb["vs_ms"].notna().all()
    assert df_comb["depth_m"].notna().all()
