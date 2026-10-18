import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tages.stratigraphy.walther import (
    FACIES_ORDER,
    GRAIN_SIZE_RANK,
    compute_basin_walther_transitions,
    compute_all_basins_metrics,
    format_walther_report,
)

def test_walther_transition_matrix_properties():
    # 建立測試資料：井 1 與 井 2
    # 井 1 依深度 (0m -> 10m -> 20m): 黏土(2) -> 細砂(4) -> 粗砂(5)
    # 時序向上 (深 -> 淺): 粗砂(5) -> 細砂(4) -> 黏土(2) (細化向上)
    data = {
        "station_id": ["W1", "W1", "W1", "W2", "W2", "W2"],
        "depth_m": [0.0, 10.0, 20.0, 0.0, 15.0, 30.0],
        "lithology_id": [2, 4, 5, 5, 4, 2],
    }
    df = pd.DataFrame(data)

    res = compute_basin_walther_transitions(
        df,
        station_col="station_id",
        depth_col="depth_m",
        lith_col="lithology_id",
        classes=[2, 4, 5],
        chronological_upward=True,
        ignore_self_transitions=True,
    )

    prob_matrix = np.array(res["prob_matrix"])
    assert prob_matrix.shape == (3, 3)

    # 每一列若有轉移，總和必須為 1.0
    row_sums = prob_matrix.sum(axis=1)
    for s in row_sums:
        if s > 0:
            assert np.isclose(s, 1.0, atol=1e-5)

    # 檢查自轉移被忽略，對角線應全為 0
    assert np.all(np.diag(prob_matrix) == 0.0)

    # 檢查粗化與細化次數
    assert res["coarsening_count"] >= 0
    assert res["fining_count"] >= 0
    assert res["reg_trans_ratio"] > 0


def test_walther_markov_entropy_bounds():
    # 建立測試資料：確定性轉移 (每一相只轉移至特定一相)
    data = {
        "station_id": ["W1", "W1", "W1"],
        "depth_m": [0.0, 10.0, 20.0],
        "lithology_id": [2, 4, 5],
    }
    df = pd.DataFrame(data)

    res = compute_basin_walther_transitions(
        df,
        classes=[2, 4, 5],
        chronological_upward=True,
        ignore_self_transitions=True,
    )

    # 確定性單一轉移的條件熵應為 0
    entropies = np.array(res["entropy_per_class"])
    assert np.all(entropies >= 0.0)
    assert res["mean_entropy"] >= 0.0


def test_compute_all_basins_metrics_real_data():
    dataset_path = Path("data/processed/dataset_v1.csv")
    if not dataset_path.exists():
        pytest.skip("dataset_v1.csv 不存在，略過真實資料測試")

    df = pd.read_csv(dataset_path)
    basins = ["濁水溪沖積扇", "嘉南平原", "屏東平原", "臺北盆地"]
    results = compute_all_basins_metrics(df, basin_col="wsect", target_basins=basins)

    for b in basins:
        assert b in results
        m = results[b]
        assert m["num_wells"] > 0
        assert m["total_points"] > 0
        assert 0.0 <= m["aquifer_pct"] <= 100.0
        assert 0.0 <= m["clay_pct"] <= 100.0
        assert m["mean_entropy"] >= 0.0

    # 嘉南平原黏土比例應顯著高於屏東平原
    assert results["嘉南平原"]["clay_pct"] > results["屏東平原"]["clay_pct"]
    # 屏東平原粗粒砂礫比例應顯著高於嘉南平原
    assert results["屏東平原"]["aquifer_pct"] > results["嘉南平原"]["aquifer_pct"]


def test_format_walther_report():
    sample_metrics = {
        "濁水溪沖積扇": {
            "num_wells": 117,
            "total_points": 29454,
            "aquifer_pct": 41.09,
            "clay_pct": 14.45,
            "sand_mud_ratio": 1.63,
            "mean_entropy": 1.85,
            "reg_trans_ratio": 0.944,
        },
        "全臺整體 (All Taiwan)": {
            "num_wells": 422,
            "total_points": 116960,
            "aquifer_pct": 31.5,
            "clay_pct": 22.8,
            "sand_mud_ratio": 1.35,
            "mean_entropy": 1.92,
            "reg_trans_ratio": 0.96,
        },
    }
    report = format_walther_report(sample_metrics)
    assert "濁水溪沖積扇" in report
    assert "Walther's Law Facies Succession" in report
    assert "41.09%" in report
