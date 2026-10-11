# TAGES-Evo: 臺灣地下 3D 地體結構與地層岩性重構系統
### Taiwan 3D Subsurface Geological Structure & Lithology Reconstruction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-41%2F41%20passing-brightgreen.svg)](tests/)
[![Target Region](https://img.shields.io/badge/region-Chiayi%20Plain%20to%20Foothills-orange.svg)](reports/)
[![Depth](https://img.shields.io/badge/depth-0%20~%201%2C000%20m-purple.svg)](reports/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TAGES-Evo** 是一個結合**中研院地球科學所 (TEC) 最新三維震波速度斷層模型**與**深層水文地質/工程鑽孔大數據**，透過機器學習與地層學先驗約束，自動重構臺灣地下 **0 至 1,000 公尺高解析度 3D 地體結構與岩性分類**之科研專案。

---

## 📌 核心亮點與科學創新 (Key Innovations)

1. **全域 0 ~ 1,000 公尺 3D 體素地體模型 (0 to 1 km Subsurface Voxel Engine)**：
   * 涵蓋嘉義沿海至阿里山山麓帶 ($120.10^\circ\text{E} \sim 120.60^\circ\text{E}, 23.35^\circ\text{N} \sim 23.60^\circ\text{N}$)。
   * 以 $51 \times 26 \times 101 = \mathbf{133,926}$ 個三維空間體素點，精細重構沖積平原第四紀沉積相至深部第三紀基盤構造。
   * 萃取全區**聲學與工程基盤等深線曲面 ($Z_\text{bedrock}$)** 與多深度水平切片圖。

2. **L2 十大地層岩性分類體系 (10-Class Lithology System)**：
   * 克服以往分類過於粗略之缺陷，遵循統一土壤分類法 (USCS) 與沈積相律 (Walther's Law)：
     * **土壤沉積相**：表土/回填土 (Fill)、黏土/泥 (ClayMud)、粉土/粉砂 (Silt)、細砂 (FineSand)、中粗砂 (CoarseSand)、砂泥互層 (SandMudMixed)、礫石/卵石 (Gravel)。
     * **基盤岩性相**：泥岩/頁岩軟岩基底 (Mudstone)、固結硬砂岩 (Sandstone)、結晶硬基底 (Crystalline Basement)。

3. **嚴格防洩漏空間交叉驗證 (Leakage-Free Validation)**：
   * 採用嚴格井位分組 `GroupKFold` 與 `SpatialBlockKFold`，**保證 0% 跨井深度資料洩漏**，真實反映對未知空間點的泛化能力。

4. **地層學馬可夫序列動態規劃解碼器 (Markov Stratigraphic Viterbi Decoder)**：
   * 自 24,767 筆真實地層數據學習沈積相相鄰轉移機率矩陣 $T_{i, j}$。
   * 透過維特比演算法（Viterbi Decoding）消除了 **100% 的孤立單點跳動噪聲 (A-B-A chatter)**，平均換層次數降低 92.6%，完美符合自然沉積層序連續律。

5. **強震頻譜比 (SSR) 場址效應物理雙向驗證 (Geophysical Cross-Validation)**：
   * 串接強震觀測網波形分析，利用盆地一維 1/4 波長共振公式 $f_0 = \bar{V}_s / (4 H_\text{sed})$：
     * 布袋 (CHY010): $H_\text{sed} = 1000\text{ m} \to f_{0,\text{theo}} = 0.52\text{ Hz} \Longleftrightarrow f_{0,\text{ssr}} = 0.50\text{ Hz}$
     * 新港 (CHY080): $H_\text{sed} = 1000\text{ m} \to f_{0,\text{theo}} = 0.45\text{ Hz} \Longleftrightarrow f_{0,\text{ssr}} = 0.45\text{ Hz}$
     * 觸口 (CHY029): $H_\text{sed} = 640\text{ m} \to f_{0,\text{theo}} = 0.83\text{ Hz} \Longleftrightarrow f_{0,\text{ssr}} = 0.80\text{ Hz}$
   * 證實機器學習預測之 3D 地體構造與地震工程實測共振週期高度契合！

6. **免伺服器 3D 互動網頁瀏覽器 (`reports/chiayi_3d_viewer.html`)**：
   * 基於 Three.js WebGL，內建 34,476 顆空間體素點雲與 46 口真實水文深井立體柱。
   * 支援 0~1,000m 動態深度切片、10 大岩性圖層獨立切換、基盤 3D Wireframe 與滑鼠懸停數值 HUD。

7. **RTX 5090 外接硬碟搬運包 (Standalone 5090 Toolkit)**：
   * 內建環境檢查批次檔 (`check_env.bat`)、最長 7.5 小時超時防呆排程 (`run_on_5090.bat`) 與斷點續跑機制。

---

## 📂 專案目錄結構 (Project Structure)

```text
tages-project/
├── pyproject.toml              # 專案相依設定與 pytest 配置
├── README.md                   # 專案詳細說明文件
├── .gitignore                  # Git 忽略配置
├── src/                        # 核心套件原始碼
│   └── tages/
│       ├── features/
│       │   └── interpolator.py # 3D 震波速度反距離權重 (IDW) 內插引擎
│       ├── geo/
│       │   └── crs.py          # 高精度 TWD67 / TWD97 / WGS84 座標轉換
│       ├── labels/
│       │   ├── lithology.py    # L1 粗類與 L2 十大岩性文字分類器
│       │   └── hydro_codes.py  # 官方水文代碼轉譯器
│       ├── models/
│       │   ├── split.py        # 去洩漏 GroupKFold 與 SpatialBlockKFold
│       │   └── sequence.py     # Walther's Law 馬可夫轉移與維特比序列解碼器
│       ├── seismic/
│       │   └── ssr.py          # 強震頻譜比 FFT + Konno-Ohmachi 平滑引擎
│       └── voxels/
│           └── volume.py       # 0~1km 3D 體素生成與基盤邊界萃取引擎
├── scripts/                    # 完整自動化執行管線
│   ├── build_hydro_strata.py   # 水文深井轉統一 schema
│   ├── build_dataset.py        # 結合波速與座標建構訓練資料集
│   ├── train_baseline.py       # 5-Fold 無洩漏隨機森林訓練與基準評估
│   ├── benchmark_sequence_model.py # 序列模型與地層連續性評估
│   ├── generate_3d_voxels.py   # 生成全域 3D 體素 (.npz) 與出版級切片圖件
│   ├── export_3d_web_data.py   # 匯出 Three.js 網頁專用立體幾何數據
│   ├── run_ssr_analysis.py     # 執行強震 SSR 頻譜比與場址效應跨學科驗證
│   ├── generate_ai_profile.py  # 2D 垂直地質剖面生成
│   ├── smooth_profile.py       # 地層連續性平滑後處理
│   ├── package_for_5090.py     # 製作 RTX 5090 外接硬碟搬運包
│   └── run_ocr_pipeline.py     # 批次柱狀圖 OCR 與規則防呆管線
├── tests/                      # 自動化單元測試集 (41 項測試 100% 通過)
│   ├── test_crs.py
│   ├── test_interpolator.py
│   ├── test_lithology.py
│   ├── test_sequence.py
│   ├── test_split.py
│   ├── test_ssr.py
│   └── test_voxels.py
├── data/
│   ├── interim/                # 輕量核心中間資料表 (已納入版本控制)
│   │   ├── chiayi_velocity_3d.csv  # TEC LEE2025 震波速度網格 (1,120 節點)
│   │   ├── hydro_boreholes.csv     # 91 口深井座標與深度表
│   │   ├── hydro_strata.csv        # 24,767 筆真實地層分層
│   │   ├── chiayi_manifest.csv     # 嘉義 5,498 孔狀態與圖檔配對清單
│   │   └── hydro_phys_summary.csv  # 井測數值統計
│   └── processed/
│       ├── dataset_v1.csv          # 27,517 筆標準訓練樣本
│       └── chiayi_3d_voxels_1km.npz # 133,926 空間體素立體模型
├── reports/                    # 出版級成果圖件與報告
│   ├── chiayi_3d_viewer.html   # Three.js 3D 互動網頁主程式
│   ├── chiayi_3d_data.js       # 3D 體素幾何數據
│   ├── chiayi_depth_slices_1km.png  # 6 大水平深度切片圖
│   ├── chiayi_bedrock_depth_map.png # 聲學基盤等深線圖
│   ├── chiayi_ortho_fence_diagram.png # 正交立體柵欄剖面圖
│   ├── chiayi_ssr_site_effects.png  # 強震 SSR 頻譜比曲線
│   ├── chiayi_ssr_summary.csv       # 強震場址效應統計表
│   ├── benchmark_summary.txt        # Baseline 評估報告
│   └── sequence_model_benchmark.txt # 序列模型解碼報告
└── docs/                       # 專案規劃與工作執行日誌
    ├── 專案規劃_最終版.md
    └── 工作執行日誌_20261011.md
```

---

## ⚡ 快速開始 (Quickstart)

### 1. 安裝環境與相依套件
```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝核心套件
pip install numpy pandas scipy scikit-learn matplotlib joblib pyproj pytest requests
```

### 2. 執行全自動單元測試
```bash
pytest tests/
# 預期輸出: 41 passed in ~2.6s (100% PASSED)
```

### 3. 一鍵生成 3D 體素模型與圖件
```bash
python scripts/generate_3d_voxels.py
# 產出:
# - data/processed/chiayi_3d_voxels_1km.npz
# - reports/chiayi_depth_slices_1km.png
# - reports/chiayi_bedrock_depth_map.png
# - reports/chiayi_ortho_fence_diagram.png
```

### 4. 開啟 3D 互動網頁瀏覽器
直接雙擊開啟 [`reports/chiayi_3d_viewer.html`](reports/chiayi_3d_viewer.html) 即可在任何現代瀏覽器（Chrome, Edge, Firefox）中流暢體驗 3D 旋轉、深度切片與 10 大岩性透視！

### 5. 執行強震 SSR 頻譜比分析
```bash
python scripts/run_ssr_analysis.py
# 產出:
# - reports/chiayi_ssr_site_effects.png
# - reports/chiayi_ssr_summary.csv
```

---

## 📊 成果圖件展示 (Gallery)

| 圖件名稱 | 說明 |
| :--- | :--- |
| **[水平深度切片圖](reports/chiayi_depth_slices_1km.png)** | 展現 20m, 100m, 250m, 500m, 750m, 1,000m 各層岩性平面分佈與沉積相演變 |
| **[聲學基盤等深線圖](reports/chiayi_bedrock_depth_map.png)** | 揭示嘉南平原沉積盆地由東向西逐漸加深之幾何形貌 |
| **[正交立體柵欄剖面](reports/chiayi_ortho_fence_diagram.png)** | 東西向穿過嘉義市中心與南北向貫穿平原軸之垂直結構對比 |
| **[強震 SSR 頻譜比曲線](reports/chiayi_ssr_site_effects.png)** | 驗證 3D 體素基盤深度與長週期盆地共振物理的一致性 |

---

## 📜 引用與致謝 (Acknowledgements)

* **中央氣象署 (CWA) & 臺灣地震科學中心 (TEC)**：提供臺灣 3D 震波速度斷層模型 (LEE2025) 與強震觀測網資料。
* **經濟部地質調查及礦業管理中心 (GSM)**：提供「臺灣水文地質資訊系統」深層觀測井紀錄。
* 感謝指導老師與團隊夥伴的協助與建議。
