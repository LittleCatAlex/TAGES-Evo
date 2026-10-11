import numpy as np
import pandas as pd
import pytest
from tages.models.sequence import MarkovStratigraphicModel

def test_markov_fit():
    # 建立合成鑽孔地層數據
    # 井 1: 泥 -> 砂 -> 礫
    # 井 2: 泥 -> 砂 -> 泥
    data = {
        "station": ["W1", "W1", "W1", "W2", "W2", "W2"],
        "top_m": [0, 10, 20, 0, 15, 30],
        "l1": [2, 3, 4, 2, 3, 2]
    }
    df = pd.DataFrame(data)
    model = MarkovStratigraphicModel(classes=[2, 3, 4], alpha_smooth=0.1)
    model.fit(df)

    assert model.fitted
    assert model.trans_mat.shape == (3, 3)
    # 每列總和必須為 1.0 (條件機率)
    row_sums = np.sum(model.trans_mat, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # 泥 (2, idx 0) 後面高機率接 砂 (3, idx 1)
    assert model.trans_mat[0, 1] > model.trans_mat[0, 0]

def test_viterbi_noise_removal():
    model = MarkovStratigraphicModel(classes=[2, 3, 4], alpha_smooth=0.1)
    # 設定自轉移機率較高 (代表厚層連續)
    model.trans_mat = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.15, 0.8]
    ])
    model.initial_probs = np.array([0.5, 0.3, 0.2])

    # 模擬 10 個深度點，原本應全為 砂 (3, idx 1)，但第 5 點有微弱雜訊被誤判為 泥 (2, idx 0)
    emission = np.full((10, 3), 0.1)
    emission[:, 1] = 0.8  # 主要是 Sand
    # 單點雜訊: 第 4 點稍微偏向 Clay (0.55 vs 0.40)
    emission[4, 0] = 0.55
    emission[4, 1] = 0.40

    # 原始 argmax 會在第 4 點產生噪點 (2)
    raw_preds = np.array([model.classes[i] for i in np.argmax(emission, axis=1)])
    assert raw_preds[4] == 2

    # 維特比解碼在 lambda_transition 作用下，應該能平滑消除該孤立單點雜訊！
    decoded_seq = model.decode_viterbi(emission, lambda_transition=2.0)
    # 解碼結果全序列應穩定為 3 (Sand)
    assert np.all(decoded_seq == 3)
