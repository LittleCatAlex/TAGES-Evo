"""地質層序連續性與馬可夫鏈序列模型 (Markov Stratigraphic Sequence & Viterbi Decoder)。

科學原理：
依據地層學沈積相定律 (Walther's Law of Facies) 與沈積層序學：
相鄰沈積相在垂直剖面上的疊置並非隨機，而是具備強烈的垂直馬可夫轉移特徵 (Markovian Transition)。
傳統隨機森林 (RF) 將每個深度體素視為獨立樣本，容易在垂直方向產生高頻「椒鹽雜訊」與非地質性的頻繁跳動。

本模組：
1. 自 24,767 筆真實水文鑽孔地層數據中，學習相鄰地層轉移機率矩陣 T_ij = P(L_{t+1} = j | L_t = i)。
2. 結合機器學習發射機率 (Emission Probs from RF) 與馬可夫轉移機率 (Transition Matrix)。
3. 使用維特比演算法 (Viterbi Algorithm) 進行全域最大後驗機率 (MAP) 深度序列解碼。
4. 有效消除垂直噪點，還原符合沈積學原理的地層厚度與層序界線。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

class MarkovStratigraphicModel:
    def __init__(self, classes: list[int] | None = None, alpha_smooth: float = 1.0):
        # 預設 L2 細分沉積物類別: [2 (ClayMud), 3 (Silt), 4 (FineSand), 5 (CoarseSand), 6 (SandMudMixed), 7 (Gravel)]
        self.classes = classes or [2, 3, 4, 5, 6, 7]
        self.k = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.alpha = alpha_smooth
        
        # 初始均勻轉移矩陣
        self.trans_mat = np.ones((self.k, self.k)) / self.k
        self.initial_probs = np.ones(self.k) / self.k
        self.fitted = False

    def fit(self, strata_df: pd.DataFrame) -> MarkovStratigraphicModel:
        """自歷史鑽孔地層數據中統計轉移矩陣。"""
        # 確保依井位與深度排序
        df = strata_df.copy()
        if "station" in df.columns and "top_m" in df.columns:
            df = df.sort_values(by=["station", "top_m"]).reset_index(drop=True)
            
        trans_counts = np.full((self.k, self.k), fill_value=self.alpha, dtype=float)
        init_counts = np.full(self.k, fill_value=self.alpha, dtype=float)
        
        station_col = "station" if "station" in df.columns else "station_id"
        lith_col = "l2" if "l2" in df.columns else ("l1" if "l1" in df.columns else "lithology_id")
        
        grouped = df.groupby(station_col)
        for _, group in grouped:
            seq = group[lith_col].values
            # 僅保留有效類別
            valid_seq = [s for s in seq if s in self.class_to_idx]
            if not valid_seq:
                continue
                
            init_idx = self.class_to_idx[valid_seq[0]]
            init_counts[init_idx] += 1.0
            
            for t in range(len(valid_seq) - 1):
                idx_curr = self.class_to_idx[valid_seq[t]]
                idx_next = self.class_to_idx[valid_seq[t + 1]]
                trans_counts[idx_curr, idx_next] += 1.0
                
        # 歸一化成為條件機率
        self.initial_probs = init_counts / np.sum(init_counts)
        self.trans_mat = trans_counts / np.sum(trans_counts, axis=1, keepdims=True)
        self.fitted = True
        return self

    def decode_viterbi(
        self,
        emission_probs: np.ndarray,
        lambda_transition: float = 1.0,
        vertical_step_m: float = 10.0
    ) -> np.ndarray:
        """維特比動態規劃序列解碼 (Viterbi MAP Sequence Decoding)。
        
        參數：
            emission_probs: (N_depth, K) 來自 RF 或分類器的類別條件機率。
            lambda_transition: 馬可夫轉移權重調節因子 (大於 0，值越大越偏好厚層連續性)。
            vertical_step_m: 垂直步長 (公尺)。
            
        回傳：
            best_seq: (N_depth,) 最優地質類別代碼序列。
        """
        n_depth, n_classes = emission_probs.shape
        assert n_classes == self.k, f"類別數不匹配: 期望 {self.k}, 得到 {n_classes}"
        
        eps = 1e-12
        # 計算對數空間數值
        log_init = np.log(self.initial_probs + eps)
        # 依垂直步長自適應轉移機率：短距離偏好維持同類 (自身轉移機率加強)
        # T_eff = (1 - decay) * I + decay * T
        decay = np.clip(vertical_step_m / 20.0, 0.1, 1.0)
        eff_trans = (1.0 - decay) * np.eye(self.k) + decay * self.trans_mat
        log_trans = np.log(eff_trans + eps) * lambda_transition
        log_emission = np.log(emission_probs + eps)

        # 動態規劃矩陣
        V = np.zeros((n_depth, self.k))
        backpointer = np.zeros((n_depth, self.k), dtype=int)

        # 初始化 (t = 0)
        V[0, :] = log_init + log_emission[0, :]

        # 前向傳播 (Forward DP)
        for t in range(1, n_depth):
            # 廣播維度: V[t-1, :] 是 (K, 1), log_trans 是 (K, K)
            # prev_scores[i, j] = V[t-1, i] + log_trans[i, j]
            prev_scores = V[t - 1, :, np.newaxis] + log_trans
            best_prev = np.argmax(prev_scores, axis=0)  # (K,)
            V[t, :] = prev_scores[best_prev, np.arange(self.k)] + log_emission[t, :]
            backpointer[t, :] = best_prev

        # 回溯 (Backtracking)
        best_path_idx = np.zeros(n_depth, dtype=int)
        best_path_idx[-1] = np.argmax(V[-1, :])

        for t in range(n_depth - 2, -1, -1):
            best_path_idx[t] = backpointer[t + 1, best_path_idx[t + 1]]

        # 轉換為真實 class_id
        best_seq = np.array([self.classes[idx] for idx in best_path_idx], dtype=int)
        return best_seq

    def get_transition_dataframe(self, class_names: dict[int, str] | None = None) -> pd.DataFrame:
        """取得轉移矩陣之格式化 DataFrame。"""
        names = [class_names.get(c, str(c)) if class_names else str(c) for c in self.classes]
        return pd.DataFrame(self.trans_mat, index=names, columns=names)
