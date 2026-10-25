# TAGES-Evo: 臺灣地下 3D 地體結構與地層岩性數位分身系統
### Taiwan 3D Subsurface Geological Digital Twin & Dynamic Seismic Site Effects Reconstruction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-82%2F82%20passing-brightgreen.svg)](tests/)
[![Target Region](https://img.shields.io/badge/region-Taiwan%20All%20Basins%20%26%20Custom%20ROI-orange.svg)](reports/)
[![Depth](https://img.shields.io/badge/depth-0%20~%201%2C000%20m-purple.svg)](reports/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TAGES-Evo (Taiwan Geological & Earthquake Simulation - Evolution)** 是一個結合**中央研究院地球科學研究所 (TEC) 最新三維震波速度斷層模型 (LEE2025)**、**經濟部地質調查及礦業管理中心 (GSMMA) 6,711 口深層水文地質與工程鑽孔大數據**，以及**交通部中央氣象署 (CWA) / 國震中心 (NCREE) EGDT 32 座強震觀測測站動力反應歷時**之科學研究系統。透過深度學習（WellboreSeqNet）與沈積相律先驗約束，自動重構臺灣主要沉積盆地與全島任意區域 **0 至 1,000 公尺之高解析度 3D 地下地體結構、地層岩性分類與長週期強震場址動力放大頻譜**。

---

## 📌 核心亮點與科學創新 (Key Innovations)

1. **全臺沉積盆地與任意自訂探索區 3D 體素數位分身 (0 ~ 1,000 m Subsurface Voxel Engine)**：
   * 內建全臺灣 6 大主要沉積盆地：**嘉南平原、濁水溪沖積扇、臺北盆地、臺中盆地、屏東平原、宜蘭平原**，並支援全島任意經緯度矩形之**動態自訂探索區域 (Dynamic Custom ROI)**。
   * 精細重構第四紀未固結沉積層至深部第三紀工程基盤（$V_s \ge 760\text{ m/s}$）與聲學基盤（$V_s \ge 1,500\text{ m/s}$）之 3D 幾何接觸面。
   * 具備動態自適應相機視距（Camera Adaptive Framing），可自動根據查詢區域之空間長寬高比計算最佳視角，杜絕視角穿插與破面現象。

2. **全島連續平滑地表高程基準切換 (Topographical Elevation Datum / MSL)**：
   * 整合全臺平滑連續地表高程模型網格（臺灣高程網格 `taiwan_elev_grid`）與 6,711 口深井實測口高。
   * 支援**地表相對深度基準 ($Z \le 0$)** 與**真實大地水準面絕對海拔基準 ($Z_\text{msl} = \text{Elev} - \text{Depth}$)** 之即時無縫切換，精確再現山麓帶高程抬升、沖積平原微地形起伏與深部構造之真實三維幾何拓撲關係。

3. **32 座 TSMIP / NCREE EGDT 強震觀測測站與動力頻譜探針**：
   * 於 3D 空間中立體錨定全臺 32 座核心強震觀測站（包含嘉南、濁水溪、臺北、臺中、屏東、宜蘭），依據幾何構造位置標註**近邊緣構造帶 ($d_\text{edge} < 6\text{ km}$, 橙色)** 與**盆地平原中心區 ($d_\text{edge} \ge 6\text{ km}$, 青色)**。
   * **一維 Haskell–Thomson 剪力波理論傳遞函數頻譜 $|H(f)|$ 即時計算**：純前端根據 3D 體素垂直波速柱（$V_s$ 與密度 $\rho$）求解波動方程，即時比對實測基盤共振主頻 $f_0$。
   * **5% 阻尼擬加速度反應譜 $S_a(T)/\text{PGA}$ 動力放大曲線**：標定長週期峰值週期 $T_\text{peak}$，揭示中高樓層建築之共振破壞風險。
   * **2016 美濃地震實測觀測驗證**：對比極端長歷時與長週期加速度放大效應（如白河 CHY015 歷時 46.2s、新營 CHY054 歷時 43.1s，長週期加速度譜放大倍率達 4.1 倍）。

4. **盆地幾何邊緣效應量化回歸 (Quantitative Basin-Edge Regression)**：
   * 建立測站共振主頻殘差對邊緣距離之線性回歸模型：
     $$\text{Residual} = -0.00885 \times d_\text{edge} + 0.180\text{ Hz}\quad (r = -0.334,\, p = 0.061)$$
   * 盆地平原中心區（$d_\text{edge} \ge 6.0\text{ km}$）殘差平均僅 $+0.002\text{ Hz}$，定量證實開放盆地內部一維水平成層波傳理論之精確性；近邊緣構造帶（$d_\text{edge} < 6.0\text{ km}$）殘差達 $+0.161\text{ Hz}$，證實盆地誘發表面波與邊界幾何干涉效應（假說 H3）。

5. **L2 十大地層岩性分類與沈積相律約束 (Walther's Law Stratigraphic Continuity)**：
   * 依據統一土壤分類法 (USCS) 與自然沉積相連續規律，建立 10 大標準岩性系統：
     * **土壤沉積層**：表土/回填土 (Fill)、黏土/泥 (Clay)、粉土/粉砂 (Silt)、細砂 (FineSand)、中粗砂 (CoarseSand)、砂泥互層 (SandMudMixed)、礫石/卵石 (Gravel)。
     * **深部基盤層**：泥岩/頁岩軟岩基底 (Mudstone)、固結硬砂岩 (Sandstone)、結晶硬基底 (Crystalline Basement)。
   * 導入地層學馬可夫動態規劃維特比解碼器（Markov-Viterbi Decoder），消除 100% 孤立點雜訊跳動。

6. **免安裝全功能 3D 數位分身門戶 (`reports/taiwan_3d_portal.html`)**：
   * 左側整合 Leaflet 2D 空間 GIS 地圖（全臺活動斷層、深井群、強震測站、盆地框線、自訂框選繪製工具）。
   * 右側搭載 Three.js WebGL 3D 渲染核心，支援 3D 立體方塊（Solid Voxels）與點雲雙模式、動態深度裁切滑桿、Z 軸垂直放大倍率調節（1x ~ 30x）、10 類岩性分層獨立透視，以及滑鼠射線檢測（Raycasting）多維度資訊浮動面板。

---

## 📂 專案目錄結構 (Project Structure)

```text
tages-project/
├── pyproject.toml                     # 專案相依套件與 pytest 設定
├── README.md                          # 專案主說明文件
├── test_and_report.bat                # 一鍵全自動測試、圖件校驗與報告更新批次檔
├── retrain_pipeline.bat               # 數據更新後之一鍵模型重新訓練與驗證管線
├── .gitignore                         # Git 版本控制忽略設定
├── src/                               # 核心演算法套件
│   └── tages/
│       ├── features/
│       │   └── interpolator.py        # 3D 震波速度反距離權重 (IDW) 內插核心
│       ├── geo/
│       │   └── crs.py                 # TWD67 / TWD97 / WGS84 空間座標轉換模組
│       ├── labels/
│       │   ├── lithology.py           # L1/L2 十大岩性分類器與正規化對照表
│       │   └── hydro_codes.py         # 官方水文地質編碼轉譯模組
│       ├── models/
│       │   ├── split.py               # 井位防洩漏 GroupKFold 與空間分塊切分
│       │   └── sequence.py            # 沈積相馬可夫轉移與維特比序列解碼演算法
│       └── voxels/
│           └── volume.py              # 3D 體素網格化與基盤等深面萃取
├── scripts/                           # 資料建構、訓練與分析指令碼
│   ├── build_taiwan_portal_data.py    # 建立全臺 6,711 口深井、32 座測站與 6 大盆地 3D 門戶整合資料包
│   ├── analyze_seismic_site_effects_deep.py # 深入分析強震場址效應、傳遞函數與美濃地震動力歷時
│   ├── show_summary.py                # 顯示模型重訓各階段產物清單與功用說明
│   ├── run_automated_tests.py         # 執行各項物理與地質一致性測試
│   └── update_manuscript_verified.py  # 嚴格驗證數據來源並同步更新成果報告書
├── tests/                             # 自動化單元測試集 (82 項測試 100% 通過)
│   ├── test_chianan.py                # 嘉南平原模型基準測試
│   ├── test_crs.py                    # 座標轉換精度測試
│   ├── test_distillation.py           # 知識蒸餾模型測試
│   ├── test_interpolator.py           # 3D 速度內插器測試
│   ├── test_lithology.py              # 岩性分類規則測試
│   ├── test_manuscript_consistency.py # 研究報告書數據前後一致性檢驗
│   ├── test_school_boreholes.py       # 學校工程鑽孔整合檢驗
│   ├── test_sequence.py               # 地層序列連續性測試
│   ├── test_split.py                  # 空間資料集切分防洩漏檢驗
│   ├── test_ssr.py                    # 強震頻譜比物理解算測試
│   ├── test_taiwan_portal.py          # 3D 門戶、連續高程基準與強震測站測試
│   ├── test_trainer.py                # 機器學習訓練器測試
│   ├── test_voxels.py                 # 體素陣列與體積解算測試
│   ├── test_walther_analysis.py       # Walther 相律地層沈積相分析測試
│   ├── test_wave_propagator.py        # 一維波動傳播 Haskell 方程數值解測試
│   ├── test_well_dataset.py           # 深井特徵資料集檢驗
│   └── test_well_seq_net.py           # WellboreSeqNet 序列模型神經網路測試
├── data/
│   ├── interim/                       # 核心中繼資料 (納入版本控制)
│   │   ├── taiwan_boreholes.csv       # 全臺 6,711 口水文深井與學校鑽孔統合清單
│   │   ├── chianan_velocity_3d.csv    # TEC LEE2025 震波速度網格 (嘉南平原)
│   │   ├── chiayi_velocity_3d.csv     # TEC LEE2025 震波速度網格 (嘉義地區)
│   │   └── hydro_boreholes.csv        # 91 口深層水文地質觀測井
│   └── processed/                     # 3D 體素立體陣列 (.npz)
│       └── chianan_3d_voxels_1km.npz  # 嘉南平原 133,926 體素標準模型
├── reports/                           # 出版級成果報告書與 3D Web 數位分身門戶
│   ├── taiwan_3d_portal.html          # 【核心】全臺 3D 地下地體結構與強震動態分身門戶
│   ├── taiwan_portal_data.js          # 全臺 3D 體素幾何、強震頻譜與連續高程網格資料包
│   ├── seismic_site_effects_deep_analysis.png # 強震場址效應深度分析圖 (圖 3-13)
│   ├── seismic_site_effects_deep_metrics.json # 32 座強震測站精確指標與頻譜參數數據庫
│   └── ...                            # 水平深度切片、基盤等深線、正交柵欄圖等圖件
└── docs/                              # 研究計畫書、進度說明書與文獻筆記
```

---

## ⚡ 快速開始 (Quickstart)

### 1. 環境配置
```bash
# 建立 Python 3.10+ 虛擬環境
python -m venv .venv

# 啟用虛擬環境
# Windows PowerShell / CMD:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 安裝所需相依套件
pip install numpy pandas scipy scikit-learn matplotlib joblib pyproj pytest requests
```

### 2. 執行全自動單元測試 (82 項測試)
```bash
# 於專案根目錄執行
pytest tests/
# 預期輸出: 82 passed (100% 通過)
```

### 3. 一鍵全自動測試與報告更新 (`test_and_report.bat`)
若需要在日常開發中驗證程式碼與測試模型準確度，可直接點擊或於終端機執行：
```cmd
test_and_report.bat
```
該批次檔會依序執行 7 大自動化步驟：環境檢查、單元測試、全臺 3D 門戶資料包重構、強震場址效應物理解算、成果說明書一致性同步、管線健康度檢查與摘要輸出。

### 4. 數據更新後重新訓練模型管線 (`retrain_pipeline.bat`)
若新增或更新了鑽孔數據、斷層線或震波速度模型，需重新訓練模型時，執行：
```cmd
retrain_pipeline.bat
```
執行過程中會清晰顯示 8 個核心執行步驟（特徵萃取、無洩漏分割、模型訓練、序列平滑、體素生成、頻譜計算、門戶打包與測試校驗），完成後將自動輸出所有產物檔案之磁碟路徑與功能用途。

### 5. 開啟 3D 數位分身門戶
直接以現代網頁瀏覽器（Google Chrome、Microsoft Edge、Mozilla Firefox 等）雙擊開啟：
```text
reports/taiwan_3d_portal.html
```
即可在純前端無伺服器環境下，暢享全臺 6 大盆地與自訂探索區之 3D 地質立體旋轉、0~1,000m 動態深度裁切、地表相對深度/絕對海拔基準無縫切換，並點擊 32 座強震測站探針即時查看 $|H(f)|$ 理論傳遞函數與 $S_a(T)/\text{PGA}$ 反應譜曲線！

---

## 📊 成果圖表與視覺化展示 (Visualizations)

| 圖件名稱 | 檔案路徑 | 說明 |
| :--- | :--- | :--- |
| **全臺 3D 地質與強震分身門戶** | [`reports/taiwan_3d_portal.html`](reports/taiwan_3d_portal.html) | 整合 2D Leaflet 地圖、3D WebGL 體素、32 測站動力頻譜抽屜與連續地形切換 |
| **強震場址效應深度分析圖** | [`reports/seismic_site_effects_deep_analysis.png`](reports/seismic_site_effects_deep_analysis.png) | 4 聯版深度圖件：邊緣距離迴歸、白河/新營美濃地震歷時、四分位加速度反應譜、構造對比 |
| **聲學與工程基盤等深線圖** | [`reports/chiayi_bedrock_depth_map.png`](reports/chiayi_bedrock_depth_map.png) | 揭示沉積盆地基盤傾角向海加深之 3D 形貌與地震波阻抗躍界面 |
| **正交立體柵欄剖面圖** | [`reports/chiayi_ortho_fence_diagram.png`](reports/chiayi_ortho_fence_diagram.png) | 東西向橫越平原與南北向貫穿盆地軸之正交地質層序對比 |
| **水平多深度岩性切片圖** | [`reports/chiayi_depth_slices_1km.png`](reports/chiayi_depth_slices_1km.png) | 展現 20m、100m、250m、500m、750m、1,000m 之沉積相垂直空間演化 |

---

## 📜 真實資料來源與學術致謝 (Data Sources & Acknowledgements)

本專案所採用之所有數據與科學模型皆源自公開客觀之國家級學術機構與官方觀測網，資料真實無虛構：

* **經濟部地質調查及礦業管理中心 (GSMMA)**：
  * 「臺灣水文地質調查資訊系統」深層水文地質觀測井鑽探紀錄與物探電測曲線。
  * 「工程地質鑽孔資訊系統 (gsmma)」全臺工程鑽探資料庫。
* **中央研究院地球科學研究所 (TEC) / 交通部中央氣象署 (CWA)**：
  * 臺灣三維震波速度斷層掃描模型（LEE2025 / Kuo-Chen et al., 2012）。
  * 臺灣強震觀測網 (TSMIP) 地表強震紀錄與 2016 美濃地震高品質加速度波形資料。
* **國家地震工程研究中心 (NCREE)**：
  * 工程地質與地動特性資料庫 (EGDT) 之場址地體分類、鑽孔剪力波速 ($V_{s30}$) 與工程基盤深度標定資料。
* 感謝指導老師與諮詢專家在研究架構、地層學原理與地震工程物理模型上的悉心指導與寶貴建議。
