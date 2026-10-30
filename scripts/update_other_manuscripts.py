"""Update docs/國際科展作品說明書_強震場址效應強化版_EAEV.md and docs/國際科展作品說明書_草稿_EAEV.md with verified metrics."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "reports" / "school_borehole_integration_metrics.json", encoding="utf-8") as f:
    sch_m = json.load(f)

# Update 強震場址效應強化版
path_se = ROOT / "docs" / "國際科展作品說明書_強震場址效應強化版_EAEV.md"
if path_se.exists():
    text = path_se.read_text(encoding="utf-8")

    text = text.replace("34,853 口工程地質鑽孔（1,056,137 筆採樣）", "39,640 口工程地質鑽孔（1,111,955 筆採樣，全臺合併達 40,063 口井、1,229,933 筆採樣）")
    text = text.replace("4,294 處（混合井）垂直孤立噪聲，混合 28,969 口井交叉驗證準確率達 55.66%（Macro F1 為 0.4591）", "4,346 處（混合井）垂直孤立噪聲，混合 30,390 口井交叉驗證準確率達 56.28%（Macro F1 為 0.4685）")
    text = text.replace("工程淺孔折外盲測達 60.06%（Macro F1 達 0.5036）", "工程淺孔折外盲測達 60.49%（Macro F1 達 0.5080，增益 +32.47%）")
    text = text.replace("15 km 空間盲測達 38.73%（增益 +2.01%）", "15 km 空間盲測達 43.22%（增益 +6.61%）")
    text = text.replace("34,853 public engineering boreholes", "39,640 public engineering boreholes (40,063 total wells)")
    text = text.replace("PG-WellboreSeqNet eliminated over 4,200 vertical noise instances, achieving 55.66% 5-fold accuracy across 28,969 wells (60.06% out-of-fold borehole accuracy; 0 noise spots).", "PG-WellboreSeqNet eliminated over 4,300 vertical noise instances, achieving 56.28% 5-fold accuracy across 30,390 wells (60.49% out-of-fold borehole accuracy; 0 noise spots).")
    text = text.replace("34,853 口工程地質鑽孔（品管排除異常後保留 28,547 口有效沉積相井、689,438 筆樣本，細分為 1,056,137 筆 1 m 深度採樣）", "39,640 口工程地質鑽孔（品管排除異常後保留 29,968 口有效沉積相井、720,684 筆樣本，細分為 1,111,955 筆 1 m 深度採樣，全臺合併 40,063 口井、1,229,933 筆採樣）")
    
    text = text.replace("融合公開工程鑽孔集 (Combined, 28,969 井)", "融合公開工程鑽孔集 (Combined, 30,390 井)")
    text = text.replace("**56.92% / 0.4603** | **+20.79% / +0.1375**", "**56.82% / 0.4590** | **+20.69% / +0.1362**")
    text = text.replace("**56.13% / 0.4475** | **+20.15% / +0.1279** | 結合 14 維物理特徵準確率達 56.13%，但垂直孤立椒鹽噪聲高達 **4,294 處**", "**56.26% / 0.4491** | **+20.29% / +0.1294** | 結合 14 維物理特徵準確率達 56.26%，但垂直孤立椒鹽噪聲高達 **4,346 處**")
    
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("| **3. 全島 5-Fold GroupKFold — PG-WellboreSeqNet**"):
            lines[i] = "| **3. 全島 5-Fold GroupKFold — PG-WellboreSeqNet** | 35.27% / 0.2665 (噪聲 0) | **56.28% / 0.4685** (噪聲 0) | **+21.01% / +0.2020** | 導入固定對數深度 $z_{\\text{log}}$、雙尺度近場先驗與同質分桶後，準確率由基準 35.27% 躍升至 **56.28%**（Macro F1 達 **0.4685**），換層頻率達 **4.0 次/100m**，且維持 **0 處孤立噪聲** |"
        elif l.startswith("| **4. 工程鑽孔本地評估 (Zero-shot"):
            lines[i] = "| **4. 工程鑽孔本地評估 (Zero-shot $\\to$ OOF — Physics RF)** | 28.08% (Zero-shot) | **60.04%** (GroupKFold OOF) | **+31.96%** | 同工區相鄰鑽孔進入訓練後，工程淺孔折外盲測精度提升 +31.96% |"
        elif l.startswith("| **5. 工程鑽孔本地評估 (Zero-shot"):
            lines[i] = "| **5. 工程鑽孔本地評估 (Zero-shot $\\to$ OOF — PG-WellboreSeqNet)** | **28.03%** (Zero-shot) | **60.49%** (GroupKFold OOF) | **+32.47%** | 融合訓練後 OOF 準確率達 **60.49%**（Macro F1 達 **0.5080**，增益達 **+32.47%**） |"
        elif l.startswith("| **6. 15 km 空間區塊盲測"):
            lines[i] = "| **6. 15 km 空間區塊盲測 (Spatial Block CV)** | 36.61% | **43.22%** | **+6.61%** | 抽離 15 km 地理區塊之跨區遠場外推準確率提升 +6.61%，增幅顯著提升，但仍受制於背景震波層析網格解析度 |"
    text = "\n".join(lines)

    text = text.replace("融合 34,853 口公開工程地質鑽孔後，模型在全臺 28,969 口有效沉積井（806,398 筆樣本）的交叉驗證準確率提升至 55.66%（Macro F1 達 0.4591，換層頻率 3.8 次/100m，孤立噪聲 0 處，徹底排除隨機森林高達 4,294 處的垂直噪聲），工程鑽孔折外盲測達 60.06%（Macro F1 達 0.5036，增益達 +32.09%），15 km 空間盲測達 38.73%（+2.01%）。", "融合 39,640 口公開工程地質鑽孔後，模型在全臺 30,390 口有效沉積井（837,644 筆樣本）的交叉驗證準確率提升至 56.28%（Macro F1 達 0.4685，換層頻率 4.0 次/100m，孤立噪聲 0 處，徹底排除隨機森林高達 4,346 處的垂直噪聲），工程鑽孔折外盲測達 60.49%（Macro F1 達 0.5080，增益達 +32.47%），15 km 空間盲測達 43.22%（+6.61%）。")
    text = text.replace("在 28,969 口混合井（806,398 筆樣本）盲測中，PG-WellboreSeqNet 點對點準確率為 55.66%（Macro F1 為 0.4591），略低於隨機森林（56.13%），但成功將隨機森林高達 4,294 處的孤立噪聲徹底消除至 0 處。同時模型換層頻率（3.8 次/100m）", "在 30,390 口混合井（837,644 筆樣本）盲測中，PG-WellboreSeqNet 點對點準確率為 56.28%（Macro F1 為 0.4685），略高於隨機森林物理特徵版（56.26%），並成功將隨機森林高達 4,346 處的孤立噪聲徹底消除至 0 處。同時模型換層頻率（4.0 次/100m）")
    text = text.replace("全面融合 GSMMA 公開之 34,853 口工程地質鑽孔並導入固定基準對數深度 $z_{\\text{log}}$ 與雙尺度近場先驗後，PG-WellboreSeqNet 在 28,547 口工程鑽孔上的折外盲測（OOF）準確率由 Zero-shot 的 27.97% 大幅提升至 60.06%（增益達 +32.09%）。然而當面對 $15\\text{ km} \\times 15\\text{ km}$ 空間區塊盲測時，準確率由 36.72% 提升至 38.73%（增益僅 +2.01%）。", "全面融合 GSMMA 公開之 39,640 口工程地質鑽孔並導入固定基準對數深度 $z_{\\text{log}}$ 與雙尺度近場先驗後，PG-WellboreSeqNet 在 29,968 口工程鑽孔上的折外盲測（OOF）準確率由 Zero-shot 的 28.03% 大幅提升至 60.49%（增益達 +32.47%）。然而當面對 $15\\text{ km} \\times 15\\text{ km}$ 空間區塊盲測時，準確率由 36.61% 提升至 43.22%（增益達 +6.61%）。")
    text = text.replace("在 28,969 口混合井（80.6 萬筆樣本）中將隨機森林產生的 4,294 處垂直孤立噪聲徹底消除至 0 處，全島 5 折準確率提升至 55.66%（Macro F1 達 0.4591，換層頻率 3.8 次/100m），工程淺孔盲測達 60.06%（Macro F1 達 0.5036，增益達 +32.09%）", "在 30,390 口混合井（83.8 萬筆樣本）中將隨機森林產生的 4,346 處垂直孤立噪聲徹底消除至 0 處，全島 5 折準確率提升至 56.28%（Macro F1 達 0.4685，換層頻率 4.0 次/100m），工程淺孔盲測達 60.49%（Macro F1 達 0.5080，增益達 +32.47%）")

    path_se.write_text(text, encoding="utf-8")
    print("Updated 強震場址效應強化版 successfully!")


# Update 草稿_EAEV
path_draft = ROOT / "docs" / "國際科展作品說明書_草稿_EAEV.md"
if path_draft.exists():
    text = path_draft.read_text(encoding="utf-8")

    text = text.replace("34,853 口公共工程地質鑽孔（含 806,398 筆深度相段）", "39,640 口公共工程地質鑽孔（含 837,644 筆深度相段，全島合併 40,063 井、1,229,933 筆採樣）")
    text = text.replace("4,294 處垂直跳躍孤立噪聲徹底消除至 0 處", "4,346 處垂直跳躍孤立噪聲徹底消除至 0 處")
    text = text.replace("We synthesized 34,853 engineering boreholes (806,398 depth intervals)", "We synthesized 39,640 engineering boreholes (837,644 depth intervals, 40,063 total wells)")
    text = text.replace("4,294 spots (in Random Forest) to zero", "4,346 spots (in Random Forest) to zero")
    text = text.replace("同化 34,853 口工程鑽孔與 422 口深層水文地質井", "同化 39,640 口工程鑽孔與 422 口深層水文地質井（全島共 40,063 口井）")
    text = text.replace("34,853 口公開工程鑽孔，涵蓋 198,725 分層，細分至 806,398 筆深度相段。", "39,640 口公開工程鑽孔，涵蓋 256,853 分層，細分至 837,644 筆有效沉積相段（總採樣 1,111,955 點，全臺合併 40,063 口井、1,229,933 筆採樣）。")
    text = text.replace("同化 34,853 口工程地質井（806,398 筆深度段）", "同化 39,640 口工程地質井（837,644 筆沉積相段，總井數 40,063 口）")

    # Table 4-1 in draft
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("| **5 折 L2 沉積相準確率"):
            lines[i] = "| **5 折 L2 沉積相準確率 (Accuracy)** | 56.82% | 56.26% | **56.28%** | 單點擬合相近（權衡層序連續性） |"
        elif l.startswith("| **Macro F1-Score (六大相平衡)**"):
            lines[i] = "| **Macro F1-Score (六大相平衡)** | 0.4590 | 0.4491 | **0.4685** | 兼顧少數類（粉土）檢出率，模型達最優平衡 |"
        elif l.startswith("| **L1 大類工程分類準確率**"):
            lines[i] = "| **L1 大類工程分類準確率** | 63.97% | 63.52% | **63.22%** | 保持高水準工程大類識別 |"
        elif l.startswith("| **工程鑽孔 Out-of-Fold 盲測準確率**"):
            lines[i] = "| **工程鑽孔 Out-of-Fold 盲測準確率** | 60.04% | 60.04% | **60.49%** | **突破 60% 大關 (Macro F1 0.5080，增益 +32.47%)** |"
        elif l.startswith("| **15 km 空間隔絕外推盲測準確率**"):
            lines[i] = "| **15 km 空間隔絕外推盲測準確率** | 36.61% | 36.61% | **43.22%** | **空間外推泛化顯著提升 +6.61%** |"
        elif l.startswith("| **垂直孤立椒鹽噪聲點數"):
            lines[i] = "| **垂直孤立椒鹽噪聲點數 (單點尖峰)** | 2,573 處 | 4,346 處 | **0 處** | **徹底歸零 (消除 100% 噪聲)** |"
        elif l.startswith("| **換層頻率 (Transitions / 100m)**"):
            lines[i] = "| **換層頻率 (Transitions / 100m)** | 3.80 次 | 3.61 次 | **3.98 次** | 符合自然沉積旋回厚度 |"
    text = "\n".join(lines)

    text = text.replace("隨機森林產生的 4,294 處「單點椒鹽噪聲」", "隨機森林產生的 4,346 處「單點椒鹽噪聲」")
    text = text.replace("多達 4,294 處高頻尖銳震盪", "多達 4,346 處高頻尖銳震盪")
    text = text.replace("模型的總體準確率由 55.66% 下降至 38.73%（雖仍優於隨機森林的 36.72%）", "模型的總體準確率由 56.28% 下降至 43.22%（雖仍優於隨機森林基準的 36.61%）")
    text = text.replace("全臺 3.4 萬口工程井資料集上實現 0% 深度洩漏之 55.66% 交叉驗證精度，徹底消除隨機森林產生之 4,294 處孤立噪聲斑塊，並將極端物理倒置率由 5.2% 壓制至 0.9%。", "全臺 4.0 萬口工程井資料集（39,640 口工程鑽孔、全島共 40,063 口井）上實現 0% 深度洩漏之 56.28% 交叉驗證精度，徹底消除隨機森林產生之 4,346 處孤立噪聲斑塊，並維持 0 處垂直孤立噪聲。")

    path_draft.write_text(text, encoding="utf-8")
    print("Updated 草稿_EAEV successfully!")
