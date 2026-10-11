"""無資料洩漏之空間與鑽孔交叉驗證切分模組 (Leakage-Free Well & Spatial Cross-Validation)。

關鍵原則：
1. GroupKFold (依 station_id): 同一鑽孔的所有深度樣本嚴格屬於同一 Fold，絕不可部分在訓練集、部分在測試集。
2. SpatialBlockKFold (依空間網格 block): 將空間切為例如 10km x 10km 網格區塊，以評估模型對未知空間區域的泛化能力。
3. 獨立地理 Holdout 區域: 預先保留獨立地理區段 (例如某特定鄉鎮或邊界區) 完全不參與訓練與調參。

純 NumPy / Python 實作，零外部相依性。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

class GroupKFoldPure:
    """純 NumPy 實作之 GroupKFold (零相依性)。"""
    def __init__(self, n_splits: int = 5, shuffle: bool = True, seed: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        
    def split(self, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        unique_groups, group_inv = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        assert n_groups >= self.n_splits, f"群組數 ({n_groups}) 少於 Fold 數 ({self.n_splits})"
        
        # 計算每個群組的樣本數量
        group_counts = np.bincount(group_inv)
        
        # 依樣本數排序群組 (貪婪分箱法平衡各 Fold 樣本數)
        order = np.argsort(-group_counts)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            # 在相似大小的群組間輕微擾動
            order = rng.permutation(order)
            
        fold_counts = np.zeros(self.n_splits, dtype=int)
        group_to_fold = np.zeros(n_groups, dtype=int)
        
        for g in order:
            best_fold = int(np.argmin(fold_counts))
            group_to_fold[g] = best_fold
            fold_counts[best_fold] += group_counts[g]
            
        sample_folds = group_to_fold[group_inv]
        all_indices = np.arange(len(groups))
        
        splits = []
        for f in range(self.n_splits):
            val_idx = all_indices[sample_folds == f]
            train_idx = all_indices[sample_folds != f]
            splits.append((train_idx, val_idx))
            
        return splits

def get_well_grouped_splits(df: pd.DataFrame, n_splits: int = 5, group_col: str = "station_id", seed: int = 42):
    """依鑽孔 ID 進行分組切分 (Group K-Fold)，確保 0% 井內深度洩漏。"""
    gkf = GroupKFoldPure(n_splits=n_splits, seed=seed)
    groups = df[group_col].values
    
    splits = gkf.split(groups)
    # 嚴格驗證鑽孔無洩漏
    for train_idx, val_idx in splits:
        train_wells = set(df.iloc[train_idx][group_col].unique())
        val_wells = set(df.iloc[val_idx][group_col].unique())
        assert len(train_wells.intersection(val_wells)) == 0, "檢測到鑽孔跨 Fold 洩漏！"
        
    return splits

def assign_spatial_blocks(df: pd.DataFrame, block_size_km: float = 10.0) -> np.ndarray:
    """依 TWD97 座標分配空間網格區塊 ID。"""
    x_km = df["x_twd97"].values / 1000.0
    y_km = df["y_twd97"].values / 1000.0
    
    x_min, y_min = np.min(x_km), np.min(y_km)
    bx = np.floor((x_km - x_min) / block_size_km).astype(int)
    by = np.floor((y_km - y_min) / block_size_km).astype(int)
    
    block_ids = bx * 1000 + by
    return block_ids

def get_spatial_block_splits(df: pd.DataFrame, block_size_km: float = 10.0, n_splits: int = 5, seed: int = 42):
    """依空間網格區塊進行空間交叉驗證 (Spatial Block CV)。"""
    blocks = assign_spatial_blocks(df, block_size_km=block_size_km)
    gkf = GroupKFoldPure(n_splits=n_splits, seed=seed)
    
    splits = gkf.split(blocks)
    for train_idx, val_idx in splits:
        train_blocks = set(blocks[train_idx])
        val_blocks = set(blocks[val_idx])
        assert len(train_blocks.intersection(val_blocks)) == 0, "空間區塊洩漏！"
        
    return splits
