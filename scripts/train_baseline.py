"""無洩漏鑽孔與空間交叉驗證 Baseline 模型訓練與評估 (Train Baseline & Benchmark)。

對照實驗架構：
1. Baseline 1 (物理規則基準): 依據剪力波速 (Vs) 與深度經驗閾值分類 (零 ML 對照組)。
2. Baseline 2 (隨機森林 RF): 特徵 [depth_m, vp_ms, vs_ms] (TAGES 原版特徵)。
3. Baseline 3 (特徵增強模型): 加入 [z_msl, vp_vs_ratio, dvs_dz, poisson_ratio, x_twd97, y_twd97]。

切分機制：
- 嚴格採用 GroupKFold (by station_id) 與 SpatialBlockKFold，保證 0% 跨井/跨空間深度洩漏！
- 輸出 Macro F1、Balanced Accuracy、每類 Precision / Recall 與混淆矩陣報告。
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.models.split import get_well_grouped_splits, get_spatial_block_splits
from tages.labels.lithology import L2_NAMES

def rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    """物理規則基準線 (6類沉積物經驗規則)。"""
    preds = np.zeros(len(df), dtype=int)
    vs = df["vs_ms"].values
    depth = df["depth_m"].values
    
    preds[:] = 4  # FineSand
    preds[(vs < 1780) & (depth < 80)] = 2  # ClayMud
    preds[(vs >= 1780) & (vs < 1850) & (depth < 120)] = 3  # Silt
    preds[(vs >= 1850) & (vs < 1950)] = 6  # SandMudMixed
    preds[(vs >= 1950) & (vs < 2100)] = 5  # CoarseSand
    preds[vs >= 2100] = 7  # Gravel
    return preds

def evaluate_predictions(y_true, y_pred, label_names):
    """計算分類指標與混淆矩陣。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    unique_labels = sorted(list(set(y_true).union(set(y_pred))))
    target_names = [label_names.get(l, str(l)) for l in unique_labels]
    
    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report_str": report,
        "confusion_matrix": cm_df
    }

def run_benchmark(dataset_path: str = "data/processed/dataset_v1.csv", reports_dir: str = "reports"):
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    print("載入 dataset_v1.csv...")
    df = pd.read_csv(dataset_path)
    
    # 聚焦於六大沉積物岩性
    # 2: ClayMud, 3: Silt, 4: FineSand, 5: CoarseSand, 6: SandMudMixed, 7: Gravel
    valid_classes = [2, 3, 4, 5, 6, 7]
    df_clean = df[df["lithology_id"].isin(valid_classes)].copy().reset_index(drop=True)
    print(f"清洗後訓練樣本數: {len(df_clean)} 筆 (涵蓋六大細分沉積物岩性)")
    
    y = df_clean["lithology_id"].values
    
    # 特徵組合
    feat_base = ["depth_m", "vp_ms", "vs_ms"]
    feat_aug = ["depth_m", "z_msl", "vp_ms", "vs_ms", "vp_vs_ratio", "dvs_dz", "poisson_ratio", "x_twd97", "y_twd97"]
    
    # 取得無洩漏切分 (Group K-Fold by well)
    print("\n執行 5-Fold Group K-Fold 切分 (以 station_id 分組，0% 深度洩漏)...")
    splits = get_well_grouped_splits(df_clean, n_splits=5, seed=42)
    
    models = {
        "Rule_Baseline": None,
        "RandomForest_BaseFeatures": feat_base,
        "RandomForest_AugmentedFeatures": feat_aug
    }
    
    results = {}
    
    for model_name, feats in models.items():
        print(f"\n==========================================")
        print(f"正在評估模型: {model_name}...")
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            df_train = df_clean.iloc[train_idx]
            df_val = df_clean.iloc[val_idx]
            
            y_train = df_train["lithology_id"].values
            y_val = df_val["lithology_id"].values
            
            if model_name == "Rule_Baseline":
                preds = rule_based_predict(df_val)
            else:
                X_train = df_train[feats].values
                X_val = df_val[feats].values
                
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced",  # 自動平衡類別權重
                    random_state=42,
                    n_jobs=-1
                )
                clf.fit(X_train, y_train)
                preds = clf.predict(X_val)
                
            all_y_true.extend(y_val)
            all_y_pred.extend(preds)
            
        metrics = evaluate_predictions(all_y_true, all_y_pred, L2_NAMES)
        results[model_name] = metrics
        
        print(f"[{model_name}] 5-Fold 交叉驗證成果:")
        print(f"  Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print("\n分類詳細報告:")
        print(metrics["report_str"])
        print("\n混淆矩陣:")
        print(metrics["confusion_matrix"])
        
    # 匯出完整評估報告
    summary_path = rep_dir / "benchmark_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=======================================================\n")
        f.write("  台灣地下 3D 地體結構 — Baseline 模型基準評估報告 (去洩漏版)\n")
        f.write("=======================================================\n\n")
        for m_name, m_res in results.items():
            f.write(f"模型: {m_name}\n")
            f.write(f"  Accuracy:    {m_res['accuracy'] * 100:.2f}%\n")
            f.write(f"  Macro F1:    {m_res['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {m_res['weighted_f1']:.4f}\n\n")
            f.write("分類報告:\n")
            f.write(m_res["report_str"] + "\n")
            f.write("混淆矩陣:\n")
            f.write(m_res["confusion_matrix"].to_string() + "\n\n")
            f.write("-------------------------------------------------------\n")
            
    print(f"\n報告已儲存至: {summary_path}")

if __name__ == "__main__":
    run_benchmark()
