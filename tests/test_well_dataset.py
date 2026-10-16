"""井身時序特徵與資料集單元測試 (Unit Tests for Sequence Features & Dataset)。"""
import numpy as np
import pandas as pd
import torch
import pytest

from tages.features.sequence_features import compute_column_sequence_features, SequenceFeatureExtractor, FEATURE_NAMES
from tages.models.dataset import WellboreSequenceDataset, collate_well_sequences

def test_compute_column_sequence_features():
    depths = np.linspace(0, 100, 51)
    z_msl = 20.0 - depths
    vp = 1800.0 + depths * 5.0
    vs = 300.0 + depths * 2.0
    vp_vs = vp / vs
    poisson = 0.25 * np.ones_like(vs)
    
    df_feats = compute_column_sequence_features(
        depths=depths,
        z_msl=z_msl,
        vp=vp,
        vs=vs,
        vp_vs=vp_vs,
        poisson=poisson,
        x_twd97=180000.0,
        y_twd97=2600000.0
    )
    
    assert len(df_feats) == 51
    assert list(df_feats.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 14
    assert not df_feats.isna().any().any(), "不應有任何 NaN 特徵值"
    assert (df_feats["depth_m"].values == depths).all()
    assert (df_feats["dvs_dz_1m"] > 0).all(), "剪力波速單調遞增時梯度應為正值"

def test_sequence_feature_extractor_and_dataset():
    # 建立兩口測試虛擬井
    records = []
    for st_id, n_pts in [("W1", 40), ("W2", 60)]:
        depths = np.linspace(1.0, float(n_pts), n_pts)
        for d in depths:
            records.append({
                "station_id": st_id,
                "depth_m": d,
                "z_msl": 15.0 - d,
                "vp_ms": 1750.0 + d * 3.0,
                "vs_ms": 250.0 + d * 1.5,
                "vp_vs_ratio": 2.1,
                "dvs_dz": 0.02,
                "poisson_ratio": 0.28,
                "x_twd97": 195000.0,
                "y_twd97": 2580000.0,
                "lithology_id": 2 if d < 20 else (4 if d < 40 else 5),
                "lithology_l1_id": 2 if d < 20 else 3
            })
    df_synth = pd.DataFrame(records)
    
    extractor = SequenceFeatureExtractor()
    extractor.fit(df_synth)
    assert extractor.fitted
    
    dataset = WellboreSequenceDataset(df_synth, extractor, l2_classes=[2, 3, 4, 5, 6, 7])
    assert len(dataset) == 2
    
    w1 = dataset[0]
    assert w1["x"].shape == (40, 14)
    assert w1["y_l2"].shape == (40,)
    assert w1["length"] == 40
    
    # 測試 Collation (動態 Padding)
    batch = collate_well_sequences([dataset[0], dataset[1]])
    x = batch["x"]
    mask = batch["mask"]
    y_l2 = batch["y_l2"]
    
    assert x.shape == (2, 60, 14)
    assert mask.shape == (2, 60)
    assert y_l2.shape == (2, 60)
    
    # 檢查 W1 (長度 40) 的前 40 步為 True，後 20 步為 False (Padding)
    assert mask[0, :40].all()
    assert (~mask[0, 40:]).all()
    # 檢查 W2 (長度 60) 全為 True
    assert mask[1, :60].all()
