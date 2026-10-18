import re
import os

input_file = r"d:\科展\tages-project\docs\國際科展作品說明書_最新進度版_EAEV.md"
mirror_file = r"d:\科展\國際科展作品說明書_最新進度版_EAEV.md"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Title and metadata block
meta_pattern = r"(# 利用物理與相律引導深度學習重建臺灣主要沉積盆地三維地質結構與強震場址效應預測\n\n> \*\*作品題目\*\*：.*?\n> \*\*版本日期\*\*：[^\n]*\n)"
meta_replacement = """# 利用物理與相律引導深度學習重建臺灣主要沉積盆地三維地質結構與強震場址效應預測

> **作品名稱**：利用物理與相律引導深度學習重建臺灣主要沉積盆地三維地質結構與強震場址效應預測  
> **英文名稱**：*Physics- and Stratigraphy-Guided Deep Learning for 3D Subsurface Reconstruction of Taiwan's Major Sedimentary Basins and Strong-Motion Site-Effect Prediction*  
> **參展科別**：地球與環境科學科（Earth and Environmental Sciences, EAEV）  
> **學科子類別**：地質科學 / 地球科學（Geosciences, GES）  
> **關鍵詞（最多三個）**：物理引導機器學習、瓦爾特相律、強震場址效應  
> **審查盲審規範**：本作品恪遵臺灣國際科展（TISF）與國際科展（ISEF）雙盲審查規範，全篇不揭露作者姓名、指導教師與就讀學校。所有數據、程式碼與圖表均為作者獨立研發、統計與繪製。  
> **版本日期**：2026-10-18（符合 TISF 實施要點附件四與 ISEF 評分規準版）  
"""

text = re.sub(meta_pattern, lambda m: meta_replacement, text, flags=re.DOTALL)

# 2. Update Abstract to TISF structured brackets
abstract_pattern = r"## 摘要（Abstract）.*?### 英文摘要（Abstract in English, ≤250 words）.*?\*\*Keywords\*\*: [^\n]*"
abstract_replacement = """## 摘要（Abstract）

### 中文摘要（約 350 字，結構化四段）
【研究目的】
臺灣主要平原與都會盆地人口密集，其淺部（0–1 km）沉積層的三維非均質岩性分佈直接控制含水層儲集骨架與強震場址共振效應。傳統地球物理面臨「全島震波層析網格過粗無法解析米級砂泥層，水文鑽井高度稀疏」之跨尺度矛盾，純數據驅動機器學習更頻繁產生違反沉積常理的孤立噪聲與逆序倒置。本研究旨在建立結合沉積相律與岩石物理先驗之物理引導序列神經網路，實現地質連續且抗噪之盆地三維高解析結構重建與場址效應正演。

【研究過程與方法】
整合全臺 422 口深層水文觀測井（116,960 筆採樣）、384 口井原位電測（656,587 筆深度點）與中研院 TEC LEE2025 全島 3D 震波速度場，建構 14 維時序與岩石物理特徵工程。提出 PG-WellboreSeqNet 模型，融合 1D-ResNet、BiLSTM 與以瓦爾特相律經驗轉移矩陣初始化之條件隨機場（CRF），並嵌入抗物理倒置懲罰項（$\\mathcal{L}_{\\text{phy}}$）。採用嚴格無井位洩漏之 5 折 GroupKFold 盲測與 15 km 空間區塊盲測；並建構一維 Haskell–Thomson 多層黏彈性傳遞函數，針對 12 座 TSMIP 測站進行強震場址共振主頻正演。

【研究結果】
在 11,640 筆盲測樣本中，模型平均準確率達 36.83%，將垂直剖面孤立單點跳變噪聲由隨機森林的 867 處徹底消除至 0 處，並將礫石誤判為軟泥之極端倒置率由 5.2% 顯著壓制至 0.9%。定量揭示六大盆地瓦爾特相界轉移熵與動力學：嘉南前陸盆地受海侵潟湖泥主導（黏土 37.02%、砂礫 6.43%、轉移熵 1.671），屏東平原與濁水溪扇則受高能水動力分選控制（粗砂礫石佔比 57.44% 與 41.13%、轉移熵 2.01–2.02）。在場址效應方面，一維傳遞函數於 12 座強震測站取得實測主頻高相關（Pearson $r = 0.9801$、RMSE = 0.233 Hz）。

【結論與應用】
本研究成功證實相律與物理先驗能有效解決地質機器學習的「椒鹽噪聲」與「物理倒置」痛點，客觀量化了點分類精度與沉積相序連續性之間的權衡取捨，為都會防震耐震微區劃、盆地邊緣效應預警與地下水資源永續評估提供具備堅實物理基礎之新範式。

**關鍵字**：物理引導機器學習、瓦爾特相律、強震場址效應

---

### 英文摘要（Abstract in English, ≤250 words）
[Objective] Subsurface 3D lithological heterogeneity in Taiwan's sedimentary basins controls regional aquifer architecture and earthquake site amplification. However, regional seismic tomography is too coarse (~km scale) to resolve metre-scale sand–mud facies, while boreholes are spatially sparse. Purely data-driven algorithms frequently produce unphysical vertical noise. We develop PG-WellboreSeqNet, a physics- and stratigraphy-guided sequence model that reconstructs high-resolution basin facies by coupling multiscale seismic petrophysics with Walther's Law of Facies.

[Methods] We integrated 422 deep hydrogeological wells (116,960 depth records), 384 wireline-logged boreholes (656,587 aligned gamma-ray and resistivity records), and the TEC LEE2025 3D velocity model across Taiwan into a 14-dimensional feature space. The architecture integrates 1D-ResNet blocks, a bidirectional LSTM, and a Linear-Chain CRF initialised with empirical facies transition priors, regularised by a rock-physics consistency loss ($\\mathcal{L}_{\\text{phy}}$). Validation was conducted under strictly zero-leakage 5-fold GroupKFold and 15-km spatial block blind tests. Forward 1D Haskell–Thomson viscoelastic transfer functions were formulated to simulate resonant frequencies at 12 TSMIP seismic stations.

[Results] In 11,640 blind test samples, PG-WellboreSeqNet achieved 36.83% accuracy across six sedimentary facies, completely eliminating vertical salt-and-pepper noise to 0 spots (compared to 867 in Random Forest) and reducing severe gravel-to-mud inversions from 5.2% to 0.9%. Quantitative Markov transition entropy revealed distinct hydrodynamics across six basins: distal Chianan is dominated by transgressive mud (37.02% clay, entropy 1.671), whereas proximal Pingtung and Choshui fans exhibit high-energy dynamics (57.44% and 41.13% coarse gravel/sand, entropy 2.01–2.02). Simulated resonance frequencies closely matched observations ($r = 0.9801$, RMSE = 0.233 Hz).

[Conclusions] This work demonstrates that stratigraphic and petrophysical priors resolve severe unphysical artefacts in geological AI, providing a physically sound framework for seismic microzonation and sustainable aquifer management.

**Keywords**: Physics-Guided Machine Learning, Walther's Law, Strong-Motion Site Effects"""

text = re.sub(abstract_pattern, lambda m: abstract_replacement, text, flags=re.DOTALL)

# 3. Chapter 1: 壹、前言
text = text.replace("## 壹、研究動機與核心科學問題", "## 壹、前言")
text = text.replace("### 1.1 研究背景與工程地質挑戰", "### 一、研究背景與工程地質挑戰")
text = text.replace("### 1.2 核心地球科學問題（Research Questions）", "### 二、核心地球科學問題（Research Questions）")
text = text.replace("### 1.3 可證偽之科學假說（Scientific Hypotheses）", "### 三、可證偽之科學假說（Scientific Hypotheses）")

# 4. Chapter 2: 貳、研究方法或過程
text = text.replace("## 貳、研究區域地質背景與多源地球物理資料", "## 貳、研究方法或過程\n\n### 一、臺灣主要沉積盆地構造分區與地質背景")
text = text.replace("### 2.1 臺灣主要沉積盆地構造分區", "#### 1.1 構造分區體系")
text = text.replace("### 2.2 多源異質地球物理資料整合", "### 二、多源異質地球物理資料整合")
text = text.replace("## 參、物理與相律引導之深度學習方法", "### 三、物理與相律引導序列深度學習（PG-WellboreSeqNet）架構")
text = text.replace("### 3.1 跨尺度物理特徵工程", "#### 3.1 跨尺度物理特徵工程")
text = text.replace("### 3.2 PG-WellboreSeqNet 神經網路架構", "#### 3.2 PG-WellboreSeqNet 神經網路架構")
text = text.replace("### 3.3 瓦爾特相律轉移先驗矩陣初始化", "#### 3.3 瓦爾特相律轉移先驗矩陣初始化")
text = text.replace("### 3.4 物理一致性損失函數（$\\mathcal{L}_{\\text{phy}}$）", "#### 3.4 物理一致性損失函數（$\\mathcal{L}_{\\text{phy}}$）")
text = text.replace("### 3.5 一維多層介質黏彈性傳遞函數正演模型", "### 四、一維多層介質黏彈性傳遞函數正演模型")

# 5. Chapter 3: 參、研究結果與討論
text = text.replace("## 肆、研究結果一：井身序列模型跨井驗證、消融實驗與混淆矩陣分析", "## 參、研究結果與討論\n\n### 一、井身序列模型跨井驗證、消融實驗與混淆矩陣分析")
text = text.replace("### 4.1 跨模型動態 5 折消融基準評估", "#### 1.1 跨模型動態 5 折消融基準評估")
text = text.replace("### 4.2 混淆矩陣與抗物理倒置分析（圖 B-2）", "#### 1.2 混淆矩陣與抗物理倒置分析（圖 3-1）")

text = text.replace("## 伍、研究結果二：臺灣六大盆地瓦爾特相律轉移矩陣與沉積動力學定量分析", "### 二、臺灣六大盆地瓦爾特相律轉移矩陣與沉積動力學定量分析")
text = text.replace("### 5.1 瓦爾特相律垂直相界轉移矩陣（圖 B-1）", "#### 2.1 瓦爾特相律垂直相界轉移矩陣（圖 3-2）")
text = text.replace("### 5.2 沉積動力學與水文地質學核心發現", "#### 2.2 沉積動力學與水文地質學核心發現")

text = text.replace("## 陸、研究結果三：井下地球物理電測約束與岩石物理模板", "### 三、井下地球物理電測約束與岩石物理模板")
text = text.replace("### 6.1 全臺 384 口觀測井電測基準分佈（圖 A-1）", "#### 3.1 全臺 384 口觀測井電測基準分佈（圖 3-3）")
text = text.replace("### 6.2 岩石物理可分離性檢驗", "#### 3.2 岩石物理可分離性檢驗")

text = text.replace("## 柒、研究結果四：三維盆地體素幾何與強震場址傳遞函數數值正演", "### 四、三維盆地體素幾何與強震場址傳遞函數數值正演")
text = text.replace("### 7.1 六大盆地三維體素重構規模", "#### 4.1 六大盆地三維體素重構規模")
text = text.replace("### 7.2 一維多層 Haskell–Thomson 強震場址主頻數值正演（12 座 TSMIP 測站）", "#### 4.2 一維多層 Haskell–Thomson 強震場址主頻數值正演（12 座 TSMIP 測站）")
text = text.replace("## 柒、研究結果四：3D 盆地幾何與強震場址效應數值正演", "### 四、3D 盆地幾何與強震場址效應數值正演")
text = text.replace("### 7.1 全臺 12 座 TSMIP 測站一維理論傳遞函數正演評估", "#### 4.1 全臺 12 座 TSMIP 測站一維理論傳遞函數正演評估")

text = text.replace("## 捌、討論：科學意義、模型局限性與反思", "### 五、綜合討論：科學意義、模型局限性與反思")
text = text.replace("### 8.1 點精確度 vs. 相序連續性之科學權衡（Trade-off Analysis）", "#### 5.1 點精確度 vs. 相序連續性之科學權衡（Trade-off Analysis）")
text = text.replace("### 8.2 空間外推邊界與大尺度層析網格之限制", "#### 5.2 空間外推邊界與大尺度層析網格之限制")
text = text.replace("### 8.3 構造基盤先驗與深層觀測缺乏之誠實聲明", "#### 5.3 構造基盤先驗與深層觀測缺乏之誠實聲明")
text = text.replace("### 8.4 強震場址正演模擬之定義界定", "#### 5.4 強震場址正演模擬之定義界定")

# 6. Chapter 4: 肆、結論與應用
text = text.replace("## 玖、結論與未來工作", "## 肆、結論與應用")
text = text.replace("### 9.1 核心結論", "### 一、核心結論")
text = text.replace("### 9.2 後續工作（Phase C & Phase D 推進路線）", "### 二、防震減災與地下水資源永續應用價值\n\n1. **都會耐震防震微區劃**：揭示沉積微相分佈與深部剛性基盤對低頻地震波放大的耦合機制，為高鐵、科學園區耐震設計提供物理導向之場址共振主頻分布預測。\n2. **地下水補注與地層下陷防制**：高解析度定量解析粗粒透水含水層（扇頂礫石、河道砂）與黏土壓密弱面，為抗旱地下水井位配置與下陷脆弱區監測提供三維分身。\n\n### 三、未來展望與後續推進路線（Phase C & Phase D）")

# 7. Chapter 5: 伍、參考文獻
text = text.replace("## 拾、主要參考文獻", "## 伍、參考文獻")

# 8. Add author self-drawn / self-made annotations to tables and figures
text = text.replace("| 編號 | 探討領域 | 核心科學問題（Research Question） | 本研究提出之檢驗方案 |", "**表 1-1 核心地球科學問題與檢驗方案對照表（本表為作者自製）**\n\n| 編號 | 探討領域 | 核心科學問題（Research Question） | 本研究提出之檢驗方案 |")
text = text.replace("| 沉積盆地名稱 | 構造體系分類 | 主導構造邊界與沉積相特徵 | 經緯度範圍（Bounding Box） |", "**表 2-1 臺灣六大代表性沉積盆地構造體系與地理邊界（本表為作者自製）**\n\n| 沉積盆地名稱 | 構造體系分類 | 主導構造邊界與沉積相特徵 | 經緯度範圍（Bounding Box） |")
text = text.replace("```mermaid\nflowchart TD\n    subgraph Data_Sources", "**圖 2-1 多源異質地球物理與地質資料庫整合流程圖（本圖為作者自繪）**\n\n```mermaid\nflowchart TD\n    subgraph Data_Sources")
text = text.replace("```mermaid\nflowchart LR\n    subgraph Inputs", "**圖 2-2 PG-WellboreSeqNet 深度學習模型總體架構圖（本圖為作者自繪）**\n\n```mermaid\nflowchart LR\n    subgraph Inputs")
text = text.replace("| 評估階段與模型名稱 | L2 沉積相 Accuracy |", "**表 3-1 井身序列模型 5 折 GroupKFold 動態基準消融評估表（本表為作者自製）**\n\n| 評估階段與模型名稱 | L2 沉積相 Accuracy |")
text = text.replace("| 沉積盆地名稱 | 鑽井數 | 總採樣點 | 粗粒含水砂礫層% |", "**表 3-2 臺灣六大沉積盆地瓦爾特相律垂直相界轉移指標統計表（本表為作者自製）**\n\n| 沉積盆地名稱 | 鑽井數 | 總採樣點 | 粗粒含水砂礫層% |")
text = text.replace("| 沉積相名稱 (L2 Facies) | 深度樣本數 | 自然伽瑪 GR 中位數", "**表 3-3 全臺 384 口水文觀測井電測岩石物理基準統計表（本表為作者自製）**\n\n| 沉積相名稱 (L2 Facies) | 深度樣本數 | 自然伽瑪 GR 中位數")
text = text.replace("| 測站代碼 (Station) | 所屬沉積盆地 | 實測 $V_{s30}$", "**表 3-4 全臺 12 座代表性 TSMIP 強震測站一維理論傳遞函數正演評估表（本表為作者自製）**\n\n| 測站代碼 (Station) | 所屬沉積盆地 | 實測 $V_{s30}$")

with open(input_file, "w", encoding="utf-8") as f:
    f.write(text)

with open(mirror_file, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully formatted TISF Attachment 4 specification. Updated both files!")
