import pandas as pd
import numpy as np
from tages.models.split import get_well_grouped_splits, assign_spatial_blocks, get_spatial_block_splits

def test_well_grouped_splits_no_leakage():
    # 建立 10 口井，每口井有 20 個深度樣本
    data = []
    for well_id in range(10):
        for depth in range(20):
            data.append({
                "station_id": f"WELL_{well_id}",
                "depth_m": float(depth),
                "x_twd97": 200000.0 + well_id * 2000.0,
                "y_twd97": 2590000.0 + well_id * 2000.0,
                "lithology_id": depth % 3
            })
    df = pd.DataFrame(data)
    
    splits = get_well_grouped_splits(df, n_splits=5)
    assert len(splits) == 5
    
    for train_idx, val_idx in splits:
        train_wells = set(df.iloc[train_idx]["station_id"])
        val_wells = set(df.iloc[val_idx]["station_id"])
        # 嚴格驗證井號無交集
        assert len(train_wells.intersection(val_wells)) == 0
        assert len(train_idx) + len(val_idx) == len(df)

def test_spatial_block_splits_no_leakage():
    data = []
    for x_offset in [0, 5, 12, 18]:  # 跨越不同的 10km 區塊
        for y_offset in [0, 15]:
            for d in range(5):
                data.append({
                    "station_id": f"W_{x_offset}_{y_offset}",
                    "depth_m": float(d),
                    "x_twd97": (200 + x_offset) * 1000.0,
                    "y_twd97": (2590 + y_offset) * 1000.0,
                    "lithology_id": 1
                })
    df = pd.DataFrame(data)
    
    splits = get_spatial_block_splits(df, block_size_km=10.0, n_splits=3)
    assert len(splits) == 3
    
    for train_idx, val_idx in splits:
        train_wells = set(df.iloc[train_idx]["station_id"])
        val_wells = set(df.iloc[val_idx]["station_id"])
        assert len(train_wells.intersection(val_wells)) == 0
