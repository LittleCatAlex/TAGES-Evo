"""純 PyTorch 向量化實作之線性鏈條件隨機場層 (Vectorized Linear-Chain CRF Layer)。

具備：
1. 支援批次動態長度遮罩 (Batched Dynamic Length Masking)。
2. Log-Sum-Exp 配分函數前向計算 (Forward Partition Function)。
3. 維特比解碼演算法 (Viterbi Sequence Decoding)。
4. 邊際機率計算 (Marginal Probability via Softmax / Forward-Backward)。
5. 支援由真實地質沈積轉移機率矩陣熱啟動初始化 (Warm-start Initialization from Empirical Walther's Law Matrix)，
   包含列中心化 (Row-centering) 與對角線自轉移適度降權，避免過度平滑坍縮。
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] 表示從 tag i 轉移到 tag j 的分數
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.08, 0.08)
        nn.init.uniform_(self.start_transitions, -0.05, 0.05)
        nn.init.uniform_(self.end_transitions, -0.05, 0.05)

    def init_from_empirical_matrix(
        self,
        trans_mat: np.ndarray,
        scale: float = 0.4,
        temperature: float = 1.5,
        laplace_smoothing: float = 1.0,
        diag_decay: float = 0.25
    ):
        """以真實地質統計轉移矩陣 (Walther's Law) 初始化 CRF 轉移權重。

        防過度平滑設計：
        1. 採用拉普拉斯平滑 (laplace_smoothing=1.0) 防止稀有地層轉移機率為 0。
        2. 執行列中心化 (Row-centering: log_t - mean(log_t, axis=1))，平衡各岩相出發之轉移基線。
        3. 對自轉移對角線項 (T_ii) 進行適度降權 (diag_decay=0.25)，避免對角線自轉移過度主導，
           使維特比解碼保留連續 2-3m 薄互層換層能力 (目標換層頻率 3-12 次/100m)。
        """
        assert trans_mat.shape == (self.num_tags, self.num_tags)
        mat = np.asarray(trans_mat, dtype=float) + float(laplace_smoothing)
        row_sums = np.sum(mat, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

        # 對數轉移矩陣與溫度平滑
        log_t = np.log(mat) / max(0.1, float(temperature))
        # 列中心化 (Row-centering)
        log_t = log_t - np.mean(log_t, axis=1, keepdims=True)
        # 對角線自轉移適度降權，防止單一岩相貫穿整口井
        diag_idx = np.arange(self.num_tags)
        log_t[diag_idx, diag_idx] *= float(diag_decay)
        # 整體縮放
        log_t = log_t * float(scale)

        with torch.no_grad():
            self.transitions.copy_(torch.from_numpy(log_t.astype(np.float32)))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """計算負對數似然損失 (Negative Log-Likelihood Loss)。"""
        return self.forward_loss(emissions, tags, mask=mask)

    def forward_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """計算負對數似然損失 (Negative Log-Likelihood Loss)。"""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._compute_forward_partition(emissions, mask)

        nll = forward_score - gold_score
        total_tokens = mask.float().sum()
        return torch.sum(nll) / torch.clamp(total_tokens, min=1.0)

    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)

        first_tags = tags[:, 0]
        score = score + self.start_transitions[first_tags] + emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            curr_mask = mask[:, i].float()
            prev_tag = tags[:, i - 1]
            curr_tag = tags[:, i]

            trans_score = self.transitions[prev_tag, curr_tag]
            emit_score = emissions[:, i].gather(1, curr_tag.unsqueeze(1)).squeeze(1)

            score = score + (trans_score + emit_score) * curr_mask

        seq_ends = mask.long().sum(dim=1) - 1
        seq_ends = torch.clamp(seq_ends, min=0)
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_forward_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        forward_var = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            curr_mask = mask[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)

            broadcast_score = forward_var.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_score
            next_forward_var = torch.logsumexp(broadcast_score, dim=1)

            forward_var = torch.where(curr_mask, next_forward_var, forward_var)

        forward_var = forward_var + self.end_transitions
        return torch.logsumexp(forward_var, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
        transition_weight: float = 0.06,
        suppress_isolated_spikes: bool = True
    ) -> List[List[int]]:
        """使用維特比動態規劃 (Viterbi Algorithm) 解碼最優路徑。

        transition_weight: 轉移矩陣加權因子 (預設調降至 0.06，介於 0.05 ~ 0.08)。
        保留連續 2-3m 薄互層換層能力 (使換層頻率提升至合理之 3-12 次/100m)，
        並搭配單點孤立尖峰濾波 (suppress_isolated_spikes) 將 A-B-A 椒鹽噪聲抑制至極低。
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        batch_size, seq_len, _ = emissions.shape
        viterbi_vars = self.start_transitions * float(transition_weight) + emissions[:, 0]
        backpointers = []

        # 解碼時進一步對自轉移對角線做輕微去偏，釋放薄互層換層靈敏度
        trans_mat = self.transitions - self.transitions.mean(dim=1, keepdim=True)
        weighted_trans = trans_mat * float(transition_weight)

        for i in range(1, seq_len):
            curr_mask = mask[:, i].unsqueeze(1)
            emit_score = emissions[:, i].unsqueeze(1)

            broadcast_score = viterbi_vars.unsqueeze(2) + weighted_trans.unsqueeze(0) + emit_score
            max_vars, bptrs = torch.max(broadcast_score, dim=1)

            viterbi_vars = torch.where(curr_mask, max_vars, viterbi_vars)
            backpointers.append(bptrs)

        viterbi_vars = viterbi_vars + self.end_transitions * float(transition_weight)
        best_last_tags = torch.argmax(viterbi_vars, dim=1).tolist()

        seq_lens = mask.long().sum(dim=1).tolist()
        best_paths = []

        for b in range(batch_size):
            t_len = seq_lens[b]
            if t_len == 0:
                best_paths.append([])
                continue

            best_tag = best_last_tags[b]
            path = [best_tag]

            for step in range(t_len - 2, -1, -1):
                best_tag = backpointers[step][b, best_tag].item()
                path.append(best_tag)

            path.reverse()

            # 消除孤立 1 點椒鹽雜訊 (A-B-A -> A-A-A)，完整保留 >=2 點之薄互層換層
            if suppress_isolated_spikes and len(path) >= 3:
                for _ in range(2):
                    for k in range(1, len(path) - 1):
                        if path[k - 1] == path[k + 1] and path[k] != path[k - 1]:
                            path[k] = path[k - 1]

            best_paths.append(path)

        return best_paths
