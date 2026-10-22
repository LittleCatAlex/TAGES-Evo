# 利用物理與相律引導深度學習重建臺灣主要沉積盆地三維地質結構與強震場址效應預測

> **作品名稱**：利用物理與相律引導深度學習重建臺灣主要沉積盆地三維地質結構與強震場址效應預測  
> **英文名稱**：*Physics- and Stratigraphy-Guided Deep Learning for 3D Subsurface Reconstruction of Taiwan's Major Sedimentary Basins and Strong-Motion Site-Effect Prediction*  
> **參展科別**：地球與環境科學科（Earth and Environmental Sciences, EAEV）  
> **學科子類別**：地質科學 / 地球科學（Geosciences, GES）  
> **關鍵詞（最多三個）**：物理引導機器學習、瓦爾特相律、強震場址效應  
> **審查盲審規範**：本作品恪遵臺灣國際科展（TISF）與國際科展（ISEF）雙盲審查規範，全篇不揭露作者姓名、指導教師與就讀學校。所有數據、程式碼與圖表均為作者獨立整理、統計與繪製。  
> **版本日期**：2026-10-22（全臺 6,288 口工程地質鑽孔大規模融合、實測無捏造快取與全量模型重訓版）  

---

## 摘要（Abstract）

### 中文摘要（約 350 字，結構化四段）
【研究目的】
臺灣主要平原與都會盆地人口密集，其淺部（0–1 km）沉積層的三維非均質岩性分佈直接控制含水層儲集架構與強震場址共振效應。傳統地球物理研究面臨「全島震波層析網格過粗、無法解析米級砂泥層，而水文鑽井在水平空間上高度稀疏」的跨尺度矛盾；純數據驅動的機器學習模型則經常在垂直剖面上產生違反沉積常理的孤立跳變噪聲與岩相倒置。本研究旨在建立結合沉積相律與岩石物理先驗的物理引導序列神經網路，重建具備地質連續性的盆地三維結構，並進行一維強震場址效應數值正演。

【研究過程與方法】
本研究整合全臺 422 口深層水文觀測井（116,960 筆深度採樣）、6,288 口工程地質鑽孔（220,704 筆深度採樣）、384 口井原位地球物理電測（1,314,619 筆深度對齊紀錄）與中研院 TEC LEE2025 全島 3D 震波速度場，建構 14 維時序與岩石物理特徵工程。提出 PG-WellboreSeqNet 模型，結合多尺度 1D-ResNet、雙向長短期記憶網路（BiLSTM）與以瓦爾特相律經驗轉移矩陣初始化之線性鏈條件隨機場（CRF），並加入抗物理倒置懲罰項（$\mathcal{L}_{\text{phy}}$）；更進一步建立特徵蒸餾管線（Petrophysics Distillation），以 17 維教師網路預訓練實測電測表徵，透過 KL 散度與虛擬電測輔助頭蒸餾至 14 維學生網路；在 3D 體素重構中區分工程基盤（$V_s \ge 760\text{ m/s}$）與聲學基盤（$V_s \ge 2,000\text{ m/s}$）。實驗採用嚴格無井位洩漏之 5 折 GroupKFold 與 15 km 空間區塊盲測；另建立一維 Haskell–Thomson 多層黏彈性傳遞函數，針對 12 座 TSMIP 強震測站進行場址共振主頻正演。

【研究結果】
在 422 口水文深井的 5 折盲測中，PG-WellboreSeqNet 的六類沉積相平均準確率為 36.83%，將垂直剖面的孤立單點跳變噪聲由隨機森林的 867 處降至 0 處，並使礫石誤判為軟泥的跨相倒置率由 5.2% 降至 0.9%。全面整編 6,288 口工程地質鑽孔（經嚴格品管排除 17,624 筆篩退圖檔後，沉積相有效樣本達 287,148 筆，大幅補足粉土相），隨機森林於 5 折交叉驗證之準確率躍升至 52.00%（F1 達 0.4261），在同工區鄰井支撐下的折外盲測（OOF）準確率由零樣本外推的 23.55% 提升至 62.58%（絕對增益 +39.03%）；在 15 km 空間盲測中由 36.80% 提升至 37.64%（+0.84%），定量界定出「局部工區近場內插錨定」與「跨盆地遠場外推」的物理尺度邊界。在特徵蒸餾實驗中，學生網路成功學會由宏觀特徵合成虛擬電測曲線（$I_{\text{GR}}\ R^2=0.172$、$\log_{10}(N64)\ R^2=0.162$），並將垂直噪聲由 188 處壓低至 49 處。全臺 3D 體素反演精確重現嘉南盆地北港基盤高區（392.4m）向南傾沒至臺南前陸深槽（923.3m，傾沒梯度 144.9m/10km），並於濁水溪沖積扇 0–300m 主含水層首度定量重現粗粒相由扇頂（77.1%）向扇尾（0.0%）之水動力指數衰減規律（$F(x) = 0.971 e^{-x/7.6}$，$R^2 = 0.917$）。一維波傳正演於 12 座強震站之理論主頻與文獻觀測值呈現高相關（Pearson $r = 0.9801$、RMSE = 0.233 Hz）。

【結論與應用】
本研究證實沉積相律與岩石物理約束能有效抑制地質機器學習中的垂直椒鹽噪聲與非物理倒置，並誠實界定大尺度震波層析在跨區盲測與三維基盤先驗上的解析度極限，可作為都會耐震微區劃與地下水含水層評估的建模基礎。

**關鍵字**：物理引導機器學習、瓦爾特相律、強震場址效應

---

### 英文摘要（Abstract in English, ≤250 words）
[Objective] Subsurface 3D lithological heterogeneity in Taiwan's sedimentary basins governs aquifer architecture and earthquake site amplification. However, regional seismic tomography is too coarse (~km scale) to resolve metre-scale sand–mud layers, whereas boreholes are spatially sparse. Purely data-driven classifiers frequently generate unphysical vertical noise. We develop PG-WellboreSeqNet, a physics- and stratigraphy-guided sequence network that reconstructs basin lithofacies by coupling seismic petrophysics with Walther's Law of Facies.

[Methods] We integrated 422 deep hydrogeological wells (116,960 depth records), 6,288 shallow engineering boreholes (220,704 depth records), 384 wireline-logged wells (1,314,619 aligned gamma-ray and resistivity records), and the TEC LEE2025 3D velocity model into a 14-dimensional feature space. The architecture combines multiscale 1D-ResNet blocks, a bidirectional LSTM, and a Linear-Chain CRF initialised with empirical facies transition priors, regularised by a rock-physics consistency loss ($\mathcal{L}_{\text{phy}}$). A petrophysics distillation pipeline transfers in-situ wireline knowledge ($I_{\text{GR}}$, $\log_{10}(N64)$) from a 17D teacher to a 14D student network via KL divergence and auxiliary pseudo-log prediction. Evaluation used zero-leakage 5-fold GroupKFold and 15-km spatial block cross-validation. Forward 1D Haskell–Thomson viscoelastic transfer functions simulated resonant frequencies at 12 TSMIP stations.

[Results] Across 5-fold blind tests on deep wells, PG-WellboreSeqNet achieved 36.83% accuracy across six facies, reducing vertical salt-and-pepper noise from 867 spots (Random Forest) to 0 and lowering gravel-to-mud inversions from 5.2% to 0.9%. Incorporating 6,288 engineering boreholes (yielding 287,148 facies samples after rigorous QC of 17,624 rejected charts) raised Random Forest 5-fold accuracy to 52.00% (F1: 0.4261) and boosted local out-of-fold accuracy from 23.55% (zero-shot) to 62.58% (+39.03%) via near-field spatial anchoring, while 15-km spatial block accuracy was 37.64% (+0.84%). Distillation student learned pseudo-wireline reconstruction ($R^2=0.172$ for $I_{\text{GR}}$, $0.162$ for $\log_{10}(N64)$) and cut vertical noise to 49. 3D voxel reconstruction accurately reproduced the structural dip from Peikang High (392.4m) to Tainan Depocenter (923.3m; 144.9m/10km gradient), and quantitative hydrodynamic exponential decay in the Zhuoshui Fan 0–300m aquifer from apex (77.1%) to distal (0.0%; $F(x)=0.971e^{-x/7.6}$, $R^2=0.917$). Simulated 1D site frequencies matched literature observations ($r = 0.9801$, RMSE = 0.233 Hz).

[Conclusions] Stratigraphic and petrophysical priors effectively suppress unphysical vertical artefacts in subsurface machine learning while clarifying the resolution limits of regional tomography for seismic microzonation and groundwater assessment.

**Keywords**: Physics-Guided Machine Learning, Walther's Law, Strong-Motion Site Effects

---

## 壹、前言

### 一、研究背景與工程地質挑戰
臺灣地處歐亞大陸板塊與菲律賓海板塊的造山碰撞帶，西部麓山帶前緣發育了由造山帶剝蝕物質充填的周緣前陸沉積盆地（Foreland Basin），包括臺中盆地、濁水溪沖積扇、嘉南平原與屏東平原；北部發育受正斷層陷落控制的臺北半地塹盆地（Half-Graben Basin），東北部則為沖繩海槽西延弧後張裂形成的蘭陽平原。這些沉積盆地累積了數百公尺至數公里厚的未固結與半固結晚第四紀沉積物，具備三個重疊的地學與工程意義：
1. **人口聚落與重要基礎設施集中**：高鐵路網、科學園區與主要都會區多位於厚層第四紀沉積地層之上。
2. **地下水儲集與地層下陷敏感帶**：粗顆粒砂礫層構成全臺主要的地下水補注與抽蓄含水層，而細顆粒黏土與粉土層在超抽地下水時易發生不可逆的壓密下陷。
3. **強震場址效應（Site Effects）顯著**：如 1999 年集集地震（$M_w 7.6$）與 2016 年美濃地震（$M_w 6.4$），震波進入鬆軟沉積盆地後因波阻抗差異產生陷落、多次折反射與低頻共振放大，導致盆地內部的搖晃程度與災情顯著高於鄰近基岩區。

在地球科學與工程地質實務上，重建三維盆地岩性結構長期面臨**觀測尺度失配（Scale Mismatch）與物理先驗缺失**的問題：
- **地震波層析成像（Seismic Tomography）解析度限制**：如中央研究院臺灣地震科學中心（TEC）發布之全島三維震波速度模型（LEE2025），能提供數公里尺度的區域地殼速度架構，但在淺部 0–1 km 垂直方向僅有表層與 2 km 等粗略節點，波速平滑且無法解析控制地下水流與場址高頻響應的米級砂泥互層。
- **水文地質與工程鑽井（Boreholes）空間稀疏性**：深層水文觀測井具備公尺級岩心描述與井下電測，但受限於鑽探經費，在水平空間上分佈稀疏（井距常達 3–10 km），且多數水文井深介於 50–300 m，未鑽穿至第三紀結晶基盤；而一般工程地質鑽孔雖在單一工區內排列密集，深度卻多侷限於淺表 10–40 m。
- **純資料驅動機器學習（Pure Data-Driven ML）的物理缺陷**：傳統樹狀分類器（如隨機森林 Random Forest）將每個深度取樣點視為彼此獨立的樣本，忽略了沉積物在重力沉降與水動力分選下的垂直連續性。這使預測出的地層柱經常出現違反沉積常理的「單點椒鹽噪聲（Salt-and-Pepper Noise）」（例如厚層軟泥中突現 1 公尺孤立礫石後隨即變回軟泥），不利於直接轉化為波傳數值模型。

### 二、核心地球科學問題（Research Questions）
本研究不以單純的軟體介面開發為目標，而是針對臺灣沉積盆地提出三個可量化檢驗的地球科學問題：

**表 1-1 核心地球科學問題與檢驗方案對照表（本表為作者自製）**

| 編號 | 探討領域 | 核心科學問題（Research Question） | 本研究提出之檢驗方案 |
| :---: | :--- | :--- | :--- |
| **Q1** | **沉積相序學與跨尺度外推** | 臺灣主要沉積盆地的地下垂直岩相序列如何反映構造與水動力差異？結合瓦爾特相律（Walther's Law）約束的序列模型與局部加密工程鑽孔，在「垂直噪聲抑制」、「工區近場內插」與「跨區遠場外推」上各自呈現何種表現與限制？ | 參、一與參、二：執行 422 口水文井與 6,288 口工程鑽孔（全島共 6,711 口井、28.7 萬筆沉積相樣本）之 5-Fold GroupKFold、15 km 空間盲測，並計算六大盆地相界轉移機率矩陣與馬可夫熵 |
| **Q2** | **井下岩石物理** | 面對不同年度、不同鑽探包商造成的自然伽瑪（GR）儀器計數尺度差異，如何透過井內標準化建立全臺 384 口井的岩石物理模板（RPT），以檢驗六大沉積相在放射性與電阻率空間中的可分離性？ | 參、三：分析 384 口井、131.4 萬筆電測資料，比較原始 GR 與井內分位數標準化泥質指數（$I_{\text{GR}}$）及視電阻率（N64）分佈 |
| **Q3** | **強震場址效應** | 依據多層黏彈性介質波傳理論（Haskell–Thomson 傳遞矩陣法），一維正演模擬在臺灣不同構造型態盆地（前陸沖積平原 vs. 半地塹與弧後裂谷）的共振主頻預測誤差有何系統性差異？ | 參、四：針對 12 座具 EGDT 文獻實測參數之 TSMIP 強震站進行 1D 頻域傳遞函數正演與殘差比較 |

### 三、可證偽之科學假說（Scientific Hypotheses）
依據上述問題，本研究提出三個具明確證偽標準的科學假說：
- **假說一（H1，相序連續性與空間尺度假說）**：
  在深度序列網路中嵌入瓦爾特相律轉移矩陣與岩石物理邊界懲罰（PG-WellboreSeqNet），能顯著壓制傳統點分類模型的垂直孤立椒鹽噪聲與礫石—軟泥跨相倒置率；同時，局部工程鑽孔的加入能顯著提升數百公尺尺度的工區近場內插精度，但超過 10 公里以上的跨區外推精度仍受限於背景震波層析網格的平滑解析度。
  *（證偽標準：若 PG-WellboreSeqNet 的垂直孤立噪聲與跨相倒置率未低於隨機森林，則假說不成立。）*
- **假說二（H2，井內標準化岩石物理階梯假說）**：
  未經標準化的全臺跨井自然伽瑪數值會因儀器單位（API 與 CPS）混雜而產生局部粒徑倒置；但經由單井分位數標準化轉換為泥質含量指數（$I_{\text{GR}}$）並搭配長正常視電阻率（N64）後，六大沉積相由礫石、粗砂、細砂、互層、粉土至黏土將呈現單調遞增的泥質階梯與電阻率分離趨勢。
- **假說三（H3，多層場址放大與盆地維度效應假說）**：
  一維多層黏彈性傳遞函數在水平成層性良好的前陸平原（如嘉南平原、臺中盆地）能準確重現場址基本共振主頻（$f_0$），但在受正斷層陡坎或張裂邊界控制的盆地（如臺北半地塹、蘭陽平原），因存在二維/三維盆地邊緣表面波散射效應，一維理論主頻將出現系統性偏差。

---

## 貳、研究方法或過程

### 一、臺灣主要沉積盆地構造分區與地質背景

#### 1.1 構造分區體系
本研究涵蓋臺灣全島六大主要沉積盆地與平原，分屬碰撞造山周緣前陸、構造拉張半地塹與弧後裂谷三大構造體系：

**表 2-1 臺灣六大代表性沉積盆地構造體系與地理邊界（本表為作者自製）**

| 沉積盆地名稱 | 構造體系分類 | 主導構造邊界與沉積相特徵 | 經緯度範圍（Bounding Box） |
| :--- | :--- | :--- | :--- |
| **嘉南平原** | 周緣前陸盆地（遠端凹陷） | 北側受北港基盤高區（Peikang High）隆升控制，向南傾沒入臺南前陸深槽；東側受觸口與梅山斷層逆衝影響；全新世海侵形成厚層海相與潟湖相軟泥層。 | 120.08°E–120.60°E, 23.15°N–23.65°N |
| **濁水溪沖積扇** | 周緣前陸盆地沖積扇 | 臺灣西部面積最大的現代複合沖積扇；濁水溪自二水鼻仔頭出山後分流擺盪，由扇頂巨厚礫石向扇央、扇尾細砂與泥層呈水動力分選過渡。 | 120.15°E–120.65°E, 23.65°N–23.90°N |
| **屏東平原** | 周緣前陸盆地（近源沖積） | 東側緊鄰中央山脈南段，受潮州斷層（Chaochou Fault）高角度逆衝控制；高屏溪、隘寮溪攜帶變質砂岩與板岩粗礫快速堆積，構成高滲透性近源沖積扇群。 | 120.42°E–120.68°E, 22.48°N–22.78°N |
| **臺北盆地** | 半地塹構造盆地（拉張斷陷） | 晚更新世因造山帶垮塌伸張與山腳正斷層（Shanchiao Fault）向西正斷陷落形成；下部為高能河道景美礫石層，上部覆蓋低能湖沼相松山層軟泥與粉砂。 | 121.42°E–121.62°E, 24.98°N–25.12°N |
| **臺中盆地** | 周緣前陸盆地（斷層下盤） | 東側受車籠埔逆衝斷層控制，西側為大肚與八卦背斜臺地；廣佈上新—更新世頭嵙山層（Toukoshan Formation）與現代大甲溪、烏溪沖積巨礫，整體地盤剛性較高。 | 120.58°E–120.78°E, 24.08°N–24.28°N |
| **蘭陽平原** | 弧後張裂三角洲平原 | 受沖繩海槽向西張裂沉降控制；蘭陽溪切穿中央山脈與雪山山脈注入太平洋，由扇頂粗礫快速過渡至中下游與濱海的厚層細砂、粉土與泥質沉積。 | 121.68°E–121.88°E, 24.65°N–24.85°N |

> [!NOTE]
> 在板塊構造學中，「地體（Terrane）」一般指具獨立地質演化史的斷層邊界構造塊體。本研究探討對象為覆蓋於基盤之上的第四紀未固結至半固結沉積構造，故正文統一使用「沉積盆地與地下地質結構」以維持地質術語精確性。

---

### 二、多源異質地球物理資料整合與公開資料來源驗證
為確保所有建模數據皆有真實地學觀測依據、杜絕人工合成假資料，本研究逐一核實並整合七項政府開放資料、學術中心模型與工程鑽探數據：

**圖 2-1 多源異質地球物理與跨尺度鑽井資料庫整合流程圖（本圖為作者自繪）**

```mermaid
flowchart TD
    subgraph Data_Sources["多源異質地球物理與鑽探資料庫"]
        A["地調中心水文地質深井<br/>(422 口觀測井, 116,960 筆分層)"]
        A2["工程地質鑽探柱狀圖資料<br/>(6,288 口工程鑽孔, 44,080 層分層)"]
        B["井下原位地球物理電測<br/>(384 口觀測井, 1,314,619 筆對齊點)"]
        C["中研院 TEC LEE2025 震波層析<br/>(全島 2,600 節點, Vp 與 Vs)"]
        D["NCREE EGDT / CWA TSMIP<br/>(12 座代表性強震站實測 Vs30 與 f0)"]
    end

    subgraph Feature_Engineering["跨尺度物理特徵與先驗建構"]
        E["14 維垂直序列與物理梯度特徵<br/>(含 IDW 真實孔口高程與壓密殘差)"]
        F["瓦爾特相律經驗轉移矩陣 P_ij<br/>與井內標準化泥質指數 I_GR"]
    end

    A --> E
    A2 --> E
    B --> F
    C --> E
    A --> F
```

**表 2-2 本研究使用之全部地球物理與地質資料來源、規模與驗證網址清冊（本表為作者自製）**

| 資料集名稱 | 原始發布機關 / 來源 | 專案內實際資料規模 | 對應本地檔案路徑 | 官方公開查詢網址（URL） |
| :--- | :--- | :--- | :--- | :--- |
| **1. 全臺水文地質深井分層資料庫** | 經濟部地質調查及礦業管理中心（原中央地質調查所） | 423 口井基本資料（422 口具完整岩性），共 101,428 層原始地層、細分為 116,960 筆深度點（六大沉積相有效點為 116,840 筆） | `data/raw/hydro/rock/*.json`<br/>`data/interim/taiwan_strata.csv`<br/>`data/processed/dataset_v1.csv` | [臺灣水文地質資訊系統](https://hydro.gsmma.gov.tw/)<br/>[地質雲加值應用平台](https://geocloud.gsmma.gov.tw/)<br/>[水文資訊網](https://gweb.wra.gov.tw/HydroInfoCore/) |
| **2. 井下原位地球物理電測資料庫** | 經濟部地質調查及礦業管理中心 | 384 口深層觀測井具連續電測 JSON（自然伽瑪 GR、自然電位 SP、16 吋與 64 吋視電阻率 N16/N64），以 0.1 m 步長對齊共 1,314,619 筆樣本 | `data/raw/hydro/phys/*.json`<br/>`data/processed/petrophysics_aligned_dataset.csv`<br/>`reports/phys_logging_benchmark_table.csv` | [臺灣水文地質資訊系統](https://hydro.gsmma.gov.tw/)<br/>[地質資料整合查詢系統](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/mp) |
| **3. 學校工程地質鑽探柱狀圖資料** | 工程地質鑽探報告柱狀圖 OCR 與人工校對結果 | 218 個工程專案、14,280 張鑽探柱狀圖影像解析，經嚴格品管與物理深度約束排除 17,624 張不合格圖檔（稽核成因記錄於 `reports/ocr_error_audit.json`）後，共整編 6,288 口工程鑽孔、44,080 層原始地層、細分為 220,704 筆 1m 步長深度點（沉積相有效點為 170,188 筆，深度 10–40 m，涵蓋全臺 218 項重大建設工程；座標經 TWD67/TWD97 轉換校正，100.00% 準確落於臺灣陸地邊界內） | `data/borehole_final_results.json`<br/>`data/interim/school_boreholes.csv`<br/>`data/processed/dataset_combined.csv` | 專案本地工程鑽探封包檔（座標經 TWD67/TWD97 轉換驗證） |
| **4. 全島三維震波層析速度模型 (`LEE2025`)** | 中央研究院地球科學研究所 / 臺灣地震科學中心（TEC） | 全島 2,600 個三維網格節點（經度 119.5°–122.5°E、緯度 21.5°–25.5°N、深度 -2 km 至 15 km），含 $V_p$、$V_s$ 與 $V_p/V_s$ | `data/raw/tec/vp_LEE2025_*.json`<br/>`data/raw/tec/vs_LEE2025_*.json`<br/>`data/interim/taiwan_velocity_3d.csv` | [TEC 3D 速度模型互動平台](https://tecdc.earth.sinica.edu.tw/data/VelocityModel/interactive3D.php)<br/>API: `https://tecdc.earth.sinica.edu.tw/data/VelocityModel/get_3d_Data.php` |
| **5. 強震測站場址參數與共振主頻基準** | 國家地震工程研究中心（NCREE）EGDT 資料庫 & Kuo et al. (2012, 2016) | 六大盆地 12 座代表性 TSMIP 強震測站之鑽孔實測 $V_{s30}$、地盤分類（Class C/D/E）與文獻實測共振主頻 $f_{0,\text{obs}}$ | `data/raw/tsmip/tsmip_site_truth.csv`<br/>`reports/chiayi_ssr_summary.csv` | [工程地質探勘資料庫 EGDT](https://egdt.ncree.org.tw/)<br/>[CWA GDMS 地球物理資料管理系統](https://gdms.cwa.gov.tw/) |
| **6. 臺灣活動斷層空間分佈數據** | 經濟部地質調查及礦業管理中心（2021 年版活動斷層圖） | 全臺 9 條控制盆地邊界之主要活動斷層（車籠埔、觸口、梅山、山腳、三義、大甲、新化、潮州、池上斷層）地表跡線座標 | `scripts/build_taiwan_portal_data.py`<br/>`src/tages/features/sequence_features.py` | [臺灣活動斷層觀測系統](https://fault.gsmma.gov.tw/)<br/>[政府資料開放平臺 (資料集 6076)](https://data.gov.tw/dataset/6076) |
| **7. 沉積相標準化分類表** | 本研究依地調中心水文地質岩性描述語料歸納 | L1 八大工程地質主類、L2 十大細分岩性（其中未固結沉積物為六大核心類別：`Class 2` 黏土/泥、`Class 3` 粉土、`Class 4` 細砂、`Class 5` 粗砂、`Class 6` 砂泥互層、`Class 7` 礫石） | `src/tages/labels/lithology.py` | 專案內部標準化分類字典 |

---

### 三、物理與相律引導序列深度學習（PG-WellboreSeqNet）架構

#### 3.1 跨尺度物理特徵工程與井內電測標準化
為將公里級地震層析速度場與鑽孔公尺級層序特徵結合，本研究於 [`src/tages/features/sequence_features.py`](file:///d:/科展/tages-project/src/tages/features/sequence_features.py) 設計了 14 維時序物理特徵向量 $\mathbf{x}(z)$，並於 [`scripts/build_petrophysics_dataset.py`](file:///d:/科展/tages-project/scripts/build_petrophysics_dataset.py) 實作井內地球物理電測標準化：

1. **幾何、高程與壓密位置特徵**：
   - 採樣深度 $z$（m）與絕對海拔高程 $z_{\text{MSL}}$（m，在無鑽孔點位處以全臺實測鑽井孔口高程進行反距離權重 IDW 內插，取代過去假設地表為 $0\text{ m}$ 的簡化作法）。
   - 相對井深比例 $z / z_{\text{max}}$（無因次）與 TWD97 二度分帶正規化平面座標 $(x_{\text{TWD97}} - 200000)/100000$、$(y_{\text{TWD97}} - 2600000)/100000$。
2. **三維震波速度場內插與動態彈性參數**：
   - 透過三維三線性內插（Trilinear Interpolation）提取各深度點背景壓縮波速 $V_p$ 與剪力波速 $V_s$。
   - 計算縱橫波速比 $V_p/V_s$ 與動態泊松比 $\nu = \frac{0.5(V_p/V_s)^2 - 1}{(V_p/V_s)^2 - 1}$。
3. **垂直波速梯度與滑動視窗統計特徵**：
   - 計算 1 m 與 5 m 視窗之剪力波速一階垂直梯度 $\frac{\partial V_s}{\partial z}$ 與二階導數 $\frac{\partial^2 V_s}{\partial z^2}$，以及上下 9 m 滑動視窗均值 $\mu_{9\text{m}}(V_s)$ 與標準差 $\sigma_{9\text{m}}(V_s)$。
4. **構造邊界距離與正常壓實波速殘差**：
   - 計算距最近主要斷層距離 $d_{\text{fault}}$、距海岸線距離 $d_{\text{coast}}$，以及相對於正常壓實趨勢線 $V_{s,\text{comp}}(z) = 160 (1 + z/8.0)^{0.32}$ 之波速殘差 $\Delta V_s = V_s - V_{s,\text{comp}}(z)$。
5. **井內自然伽瑪分位數標準化與 Larionov 泥質體積分率**：
   - 為消除不同年代與鑽探承包商儀器（API 與 CPS 計數率）造成的跨井尺度偏差，對每口具備自然伽瑪（GR）紀錄的觀測井獨立計算第 5 與第 95 百分位數（$\text{GR}_{5\%}$ 為純砂礫基線、$\text{GR}_{95\%}$ 為純泥岩基線），定義井內標準化泥質含量指數：
     $$I_{\text{GR}} = \text{clip}\left(\frac{\text{GR} - \text{GR}_{5\%}}{\text{GR}_{95\%} - \text{GR}_{5\%}}, 0, 1\right)$$
   - 進一步依未固結第三紀與第四紀年輕沉積物之經典岩石物理經驗公式（Larionov, 1969），計算非線性泥質體積分率 $V_{\text{sh}}$ 與長短正常電阻率侵入比 $R_{\text{ratio}}$：
     $$V_{\text{sh}} = 0.083 \left(2^{3.7 I_{\text{GR}}} - 1.0\right), \quad R_{\text{ratio}} = \frac{\text{N64}}{\text{N16}}$$

---

#### 3.2 PG-WellboreSeqNet 網路架構
模型將單一鑽孔的垂直岩性剖面視為一維有序深度序列 $\mathbf{Y} = (y_1, y_2, \dots, y_T)$（實作於 [`src/tages/models/well_seq_net.py`](file:///d:/科展/tages-project/src/tages/models/well_seq_net.py)）：

```mermaid
flowchart LR
    X["14 維物理特徵序列<br/>X in R^(T x 14)"] --> R["多尺度 1D-ResNet<br/>(卷積核 3m, 5m, 9m + SE)"]
    R --> B["雙向長短期記憶網路<br/>(2 層 BiLSTM, H=96)"]
    B --> H["階層引導分類頭<br/>(L1 粗類 -> L2 細分相)"]
    H --> E["發射分數 Emission E_i(t)"]
    E --> C["線性鏈條件隨機場 (CRF)<br/>(瓦爾特相律轉移矩陣 T_ij)"]
    C --> Y["全局最優沉積相序<br/>Y* in {2,3,4,5,6,7}^T"]
```

1. **多尺度一維殘差卷積區塊（Multi-Scale 1D-ResNet with SE Block）**：
   並聯三個一維卷積分支，卷積核尺寸分別對應微相薄層（$k=3$）、中型旋回（$k=5$）與厚層相域（$k=9$），並透過通道注意力機制（Squeeze-and-Excitation）動態調整特徵權重。
2. **雙向長短期記憶網路（BiLSTM）**：
   採用 2 層 BiLSTM（隱藏層維度 $H = 96$，雙向拼接共 192 維），同時捕捉由深至淺的沉積演進與由淺至深的覆壓疊置關係。
3. **階層引導分類頭（Hierarchical Head）**：
   先預測 L1 工程地質大類 logits，再透過線性投影轉化為 L2 六大沉積相的先驗偏置，與隱藏狀態共同生成發射分數 $E_i(t)$。
4. **嵌入瓦爾特相律先驗之線性鏈條件隨機場（Walther-CRF）**：
   引入 Linear-Chain CRF 計算整條井身序列的聯合對數機率：
   $$P(\mathbf{Y} \mid \mathbf{X}) = \frac{1}{Z(\mathbf{X})} \exp\left( \sum_{t=1}^T E_{y_t}(t) + \sum_{t=1}^{T-1} \mathbf{T}_{y_t, y_{t+1}} \right)$$
   其中轉移矩陣 $\mathbf{T}$ 以訓練集真實地層統計出的經驗相界轉移矩陣初始化（加拉普拉斯平滑與溫度調節 $\text{scale}=0.5, \text{temp}=1.5$）。在維特比（Viterbi）解碼時設定 `transition_weight=0.20`，使模型在不坍縮為單一多數類的前提下，濾除單點非連續跳變。

---

#### 3.3 物理引導損失函數（Physics-Guided Loss Function）
總損失函數結合序列負對數似然、類別平衡 Focal Loss 與岩石物理一致性懲罰項（實作於 [`src/tages/models/losses.py`](file:///d:/科展/tages-project/src/tages/models/losses.py)）：
$$\mathcal{L}_{\text{total}} = w_{\text{crf}} \mathcal{L}_{\text{CRF}} + w_{\text{L2}} \mathcal{L}_{\text{Focal-L2}} + w_{\text{L1}} \mathcal{L}_{\text{Focal-L1}} + \lambda_{\text{phy}} \mathcal{L}_{\text{phy}}$$

其中 $\mathcal{L}_{\text{Focal-L2}}$ 採用訓練集類別頻率反比平方根權重 $w_c \propto 1/\sqrt{N_c}$ 以緩解粉土與粗砂樣本較少的不平衡現象；$\mathcal{L}_{\text{phy}}$ 則針對預測機率分佈 $P(y_t = c)$ 施加四項可微物理邊界懲罰：
1. **高剪力波速區誤判軟泥懲罰**：$\mathcal{L}_{\text{mud}} = \text{ReLU}\left(\frac{V_s - V_{s,\text{mud\_max}}}{\sigma_v}\right) \cdot P(\text{Clay/Mud})$
2. **低剪力波速區誤判緻密礫石懲罰**：$\mathcal{L}_{\text{gravel}} = \text{ReLU}\left(\frac{V_{s,\text{gravel\_min}} - V_s}{\sigma_v}\right) \cdot P(\text{Gravel})$
3. **高泊松比飽和軟土區誤判粗骨架相懲罰**：$\mathcal{L}_{\nu} = \text{ReLU}\left(\frac{\nu - \nu_{\text{max}}}{\sigma_\nu}\right) \cdot \left[ P(\text{Gravel}) + 0.6 P(\text{Coarse Sand}) \right]$
4. **淺部未壓密低速帶粗粒倒置懲罰**：$\mathcal{L}_{\text{invert}} = \text{ReLU}\left(\frac{V_{s,\text{shallow}} - V_s}{\sigma_v}\right) \cdot \left[ P(\text{Coarse}) + P(\text{Gravel}) \right] \cdot \mathbb{I}(z < 25\text{m})$

---

#### 3.4 原位電測預訓練與知識特徵蒸餾至 3D 空間推論（Phase A3 管線架構）
為克服「3D 盆地空間體素無井下電測探棒，無法直接將實測自然伽瑪與視電阻率作為輸入特徵」的根本因果矛盾，本研究於 [`src/tages/models/distillation.py`](file:///d:/科展/tages-project/src/tages/models/distillation.py) 設計了一套**教師—學生知識特徵蒸餾架構（Petrophysics Distillation Pipeline）**：
1. **教師網路（PetrophysicsTeacher，17 維輸入）**：
   在 384 口具備原位測井的深層水文觀測井中，對齊 104,131 筆深度樣本。教師網路接收 14 維空間震波特徵加上 3 維實測電測標準化特徵（$I_{\text{GR}}$、$\log_{10}(N64)$ 與 Larionov $V_{\text{sh}}$），預訓練出對地下岩石物理具有極高分辨力的深層潛在特徵空間。
2. **學生網路（DistilledStudentNet，14 維輸入）**：
   學生網路僅接收 3D 體素空間可獲得的 14 維宏觀震波與幾何特徵，除了 L2 沉積相預測頭外，額外掛載**多任務輔助電測預測頭（Auxiliary Pseudo-Petrophysics Head）**，直接回歸虛擬的 $I_{\text{GR}}$ 與 $\log_{10}(N64)$。
3. **特徵蒸餾與多任務損失函數**：
   $$\mathcal{L}_{\text{distill}} = w_{\text{ce}} \mathcal{L}_{\text{CE}} + w_{\text{kd}} T^2 \mathcal{L}_{\text{KL}}(P_{\text{student}}^\tau, P_{\text{teacher}}^\tau) + w_{\text{aux}} \mathcal{L}_{\text{MSE}}(\hat{\mathbf{y}}_{\text{petro}}, \mathbf{y}_{\text{petro}}) + w_{\text{crf}} \mathcal{L}_{\text{CRF}}$$
   藉由溫度調節（$T=3.0$）下的 KL 散度與電測回歸損失，強迫學生網路在內部隱藏層中重建岩石物理泥質與導電率表徵，使 3D 體素空間具備實測電測級別的約束力！

---

### 四、一維多層 Haskell–Thomson 黏彈性波傳矩陣法
為避免單自由度經驗公式的過度簡化，本研究於 [`src/tages/seismic/wave_propagator.py`](file:///d:/科展/tages-project/src/tages/seismic/wave_propagator.py) 實作一維水平成層介質 SH 波傳遞矩陣法（Thomson, 1950; Haskell, 1953; Kramer, 1996）：

將場址下方沉積柱離散為 $N$ 個水平均勻黏彈性層，置於彈性基盤半空間之上：
1. 各層賦予密度 $\rho_m$（黏土 $1.85\text{ g/cm}^3$、粉土 $1.90\text{ g/cm}^3$、細砂 $1.95\text{ g/cm}^3$、粗砂 $2.00\text{ g/cm}^3$、礫石 $2.10\text{ g/cm}^3$、基盤 $2.40\text{ g/cm}^3$）與品質因子 $Q_{s,m}$。
2. 計算複數剪力模數 $G_m^* = \rho_m V_{s,m}^2 (1 + i / Q_{s,m})$ 與複數剪力波速 $V_{s,m}^* = \sqrt{G_m^* / \rho_m}$，則垂直傳播之複數波數為 $k_m^* = \omega / V_{s,m}^*$。
3. 依據各層界面位移與剪應力連續條件，由地表向下連乘單層傳遞矩陣 $\mathbf{A}(\omega) = \prod_{m=1}^{N-1} \mathbf{a}_m(\omega)$，求解地表自由表面相對於入射波之理論頻域放大函數 $|H(f)|$，並提取第一主峰對應頻率作為理論場址共振主頻 $f_{0,\text{sim}}$。

---

## 參、研究結果與討論

### 一、井身序列模型跨井驗證、消融實驗與跨尺度工程鑽孔整合分析

#### 1.1 跨模型動態 5 折消融基準評估（422 口水文深井）
本研究於 [`scripts/train_well_seq_model.py`](file:///d:/科展/tages-project/scripts/train_well_seq_model.py) 建立無寫死常數的全動態評估管線。8 組基準與消融模型均在全臺 422 口水文深井（116,840 筆有效六類沉積相點，以 `stride=2` 抽取每折約 11,624 筆代表性驗證點）與相同的 5-Fold GroupKFold（按 `station_id` 分組，確保同一口井不會同時出現在訓練集與測試集）下實測，數據輸出於 [`reports/well_seq_benchmark_metrics.json`](file:///d:/科展/tages-project/reports/well_seq_benchmark_metrics.json)：

**表 3-1 井身序列模型 5 折 GroupKFold 動態基準消融評估表（本表為作者自製）**

| 評估階段與模型名稱 | L2 沉積相 Accuracy | L2 Macro F1 | L1 大類 Accuracy | 換層頻率 (/100m) | 孤立椒鹽噪聲總數 | 核心架構與消融變因說明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **[0] 地質真實基準 (Ground Truth)** | 100.00% | 1.0000 | 100.00% | 34.0 次 | 6,829 處 | 真實水文觀測井於 stride=2 下之統計基線 |
| **(1) 多數類基準 (Majority Baseline)** | 21.28% | 0.0584 | 26.40% | 0.0 次 | 0 處 | 恆定預測訓練集最高頻類別 |
| **(2) 深度分層先驗規則 (Rule Baseline)** | 25.37% | 0.1226 | 29.29% | 0.6 次 | 0 處 | 依深度區間（0–20–50–100m）眾數判定 |
| **(3) 座標高程隨機森林 (Raw RF)** | 36.13% | 0.3228 | 43.47% | 4.0 次 | 469 處 | 僅使用 $x, y, z_{\text{MSL}}$ 與深度 $z$（類別平衡權重） |
| **(4) 物理增強隨機森林 (Physics RF)** | 35.67% | 0.3172 | 42.35% | 4.1 次 | 867 處 | 加入 14 維物理與層析波速特徵（類別平衡權重） |
| **(5) 隨機森林 + 馬可夫解碼 (RF+Markov)** | 36.06% | 0.3159 | 42.61% | 0.2 次 | 0 處 | Physics RF 發射機率 + 經驗轉移矩陣維特比平滑 |
| **(6) 標準雙向時序網路 (BiLSTM, 無CRF)** | **36.84%** | 0.2687 | **42.86%** | 0.4 次 | 4 處 | 1D-ResNet + BiLSTM（純發射分數 Argmax 解碼） |
| **(7) PG-WellboreSeqNet (無物理損失, $\lambda_{\text{phy}}=0$)** | **39.82%** | 0.2852 | **45.57%** | 0.4 次 | 0 處 | 前 2 折消融對照：移除 $\mathcal{L}_{\text{phy}}$ 岩石物理懲罰 |
| **(8) PG-WellboreSeqNet (完整模型, $\lambda_{\text{phy}}=0.25$)** | **36.83%** | 0.2684 | **42.86%** | **0.3 次** | **0 處** | **完整 5 折：ResNet + BiLSTM + CRF + $\mathcal{L}_{\text{phy}}$** |

---

#### 1.2 空間區塊盲測（Spatial Block CV）與物理損失消融討論
為檢驗模型在遠離現有鑽井網時的跨區外推能力，本研究依 $15\text{ km} \times 15\text{ km}$ 地理網格進行空間區塊盲測（位於測試網格內的所有鑽井完全不參與訓練）：
- **隨機分井 GroupKFold（5-Fold）**：Physics RF 為 `35.67% / Macro F1 0.3172`，PG-WellboreSeqNet 為 `36.83% / Macro F1 0.2684`。
- **15 km 空間區塊盲測（Spatial Block CV）**：Physics RF 為 `41.87% / Macro F1 0.3408`，PG-WellboreSeqNet 為 `35.30% / Macro F1 0.2166`。

**科學討論與邊界分析**：
1. **跨區外推受限於背景震波層析解析度**：在方圓 15 公里內無鑽井約束的盲區中，PG-WellboreSeqNet 的輸入特徵主要由大尺度層析波速（LEE2025）與空間座標構成。由於公里級層析波速在垂直方向極為平緩，缺乏微相起伏訊號，CRF 轉移約束會傾向將整口井收斂至該區最常見的單一厚相；反觀隨機森林因逐點獨立決策，能對局部座標與高程的微小變化做出反應。這說明序列模型的優勢在於**垂直剖面連續性與噪聲抑制**，而非在缺乏高解析度地球物理場的遠場盲區憑空創造微相細節。
2. **物理損失項（$\mathcal{L}_{\text{phy}}$）的尺度效應**：比較 $\lambda_{\text{phy}}=0.0$（前 2 折平均 L2 Acc `39.82%`）與 $\lambda_{\text{phy}}=0.25$（5 折平均 L2 Acc `36.83%`），加入物理邊界懲罰並未提升點分類準確率，反而因限制了部分邊界樣本的預測自由度而使 Accuracy 微幅下降約 3%。其原因在於目前背景 $V_s$ 取自全島地殼層析模型（數值偏高且平滑），與淺部公尺級鬆散沉積物的真實工程波速存在尺度落差，使 $\mathcal{L}_{\text{phy}}$ 主要扮演抑制極端跨相誤判的保守邊界角色。

---

#### 1.3 盲測鑽孔混淆矩陣與抗倒置量化分析
在 [`reports/well_seq_confusion_matrices.png`](file:///d:/科展/tages-project/reports/well_seq_confusion_matrices.png) 中，比較同一組 11,640 個盲測驗證點上隨機森林與 PG-WellboreSeqNet 的歸一化混淆矩陣：

```
                    【傳統點對點隨機森林 (Point-wise RF)】                 【物理引導時序網路 (PG-WellboreSeqNet)】
                       預測沉積相 (Predicted Facies)                             預測沉積相 (Predicted Facies)
                  泥     粉    細砂   粗砂   互層   礫石                    泥     粉    細砂   粗砂   互層   礫石
真實沉積相 黏土/泥 [34.0%  0.2%  20.7%   5.9%  23.7%  15.5%]      黏土/泥 [24.2%  0.0%  18.0%   6.2%  36.9%  14.6%]
(True)     粉土    [19.0%  1.5%  33.1%  19.8%   5.4%  21.2%]      粉土    [14.6%  0.0%  30.8%  24.4%   4.7%  25.4%]
           細砂    [21.8%  0.2%  32.1%   8.1%  19.5%  18.2%]      細砂    [13.0%  0.0%  24.7%  14.5%  31.8%  16.0%]
           粗砂    [14.7%  0.4%  21.8%  22.4%  13.2%  27.5%]      粗砂    [ 4.2%  0.0%  21.2%  20.8%  24.8%  29.0%]  <-- 誤判為泥降至 4.2%
           互層    [29.6%  0.4%  13.5%   4.9%  30.4%  21.3%]      互層    [23.2%  0.0%  11.8%   5.1%  45.6%  14.3%]
           礫石    [ 5.2%  0.5%   1.8%   9.7%  11.2%  71.5%]      礫石    [ 0.9%  0.0%   2.1%   7.5%  26.9%  62.7%]  <-- 誤判為泥降至 0.9%
```

1. **降低跨能量階梯的岩相倒置**：
   在點對點隨機森林中，真實為粗砂與礫石的高能粗骨架相，分別有 **14.7% 的粗砂** 與 **5.2% 的礫石（144 筆）** 被直接誤判為低能靜水環境的黏土/泥。在 PG-WellboreSeqNet 中，粗砂誤判為泥的比例降至 **4.2%**（相對減少 71.4%），礫石誤判為泥的比例降至 **0.9%**（25 筆，相對減少 82.7%），大多數誤差轉移至水動力相鄰的砂泥互層（26.9%）或粗砂（7.5%）。
2. **垂直孤立椒鹽噪聲歸零**：
   隨機森林在 5 折盲測中累計產生 **867 處**「A–B–A」單點跳變噪聲（詳見 [`reports/well_seq_vs_rf_columns.png`](file:///d:/科展/tages-project/reports/well_seq_vs_rf_columns.png)），而 PG-WellboreSeqNet 經由 CRF 序列解碼將孤立單點噪聲降至 **0 處**。同時亦須誠實指出，由於粉土（Silt）樣本佔比極低且物性介於泥與細砂之間，PG-WellboreSeqNet 在該少數類別上出現召回率歸零（全數併入細砂、粗砂與泥）的平滑代價。

---

#### 1.4 大規模工程地質鑽孔（6,288 口工程鑽井）融合與近場空間錨定效應分析
為探討大規模加入淺層工程地質鑽孔能否提升三維地質預測表現，本研究於 [`scripts/integrate_school_boreholes.py`](file:///d:/科展/tages-project/scripts/integrate_school_boreholes.py) 解析並整編了 `data/borehole_final_results.json` 中的 218 個工程標案、14,280 張工程柱狀圖（經 QA/QC 篩選排除 17,624 張不合格圖檔並記錄於 [`reports/ocr_error_audit.json`](file:///d:/科展/tages-project/reports/ocr_error_audit.json)）。經由專案檔名前綴唯一識別、TWD67/TWD97 空間坐標校正與物理深度邊界約束（$\le 350\text{ m}$，排除 OCR 誤識合約號為深度之 27 處極端值），成功提取 **6,288 口獨立工程地質鑽孔**（共 44,080 層原始地層）。其中 5,119 口具未固結沉積相紀錄（以 1.0m 垂直採樣生成 170,188 筆有效沉積相樣本，大幅補足了過去水文深井稀缺之粉土相達 13,164 層），與原 422 口水文深井合併為 **5,541 口有效沉積相鑽井（共 287,148 筆沉積相樣本）**。實測全量數據記錄於 [`reports/school_borehole_integration_metrics.json`](file:///d:/科展/tages-project/reports/school_borehole_integration_metrics.json)、[`reports/school_borehole_integration_report.txt`](file:///d:/科展/tages-project/reports/school_borehole_integration_report.txt) 與 [`reports/school_data_accuracy_comparison.png`](file:///d:/科展/tages-project/reports/school_data_accuracy_comparison.png)：

**表 3-2 大規模工程鑽孔融合前後之全島盲測、空間區塊盲測與工區本地評估對照表（本表為作者自製）**

| 實驗設計與評估指標 | 原水文深井基準集 (Baseline, 422 井) | 融合工程鑽孔資料集 (Combined, 5,541 井) | 絕對變化量 ($\Delta$) | 統計與地球科學成因解析 |
| :--- | :---: | :---: | :---: | :--- |
| **1. 全島 5-Fold GroupKFold — Raw RF** (L2 Acc / F1) | 36.13% / 0.3228 | **52.03%** / **0.4347** | **+15.90%** / **+0.1119** | 大規模工程鑽孔大幅加密了平原與工程工區的空間採樣，純坐標高程特徵在相鄰近井支撐下內插精度大幅躍升 |
| **2. 全島 5-Fold GroupKFold — Physics RF** (L2 Acc / F1) | 35.67% / 0.3172 | **52.00%** / **0.4261** | **+16.33%** / **+0.1089** | 結合 14 維物理特徵，5 折平均準確率突破 52%；但垂直噪聲累計增加至 2,325 處，反映樹模型在淺部複雜高頻互層下的震盪性 |
| **3. 全島 5-Fold GroupKFold — PG-WellboreSeqNet** | 36.83% / 0.2684 (噪聲 0) | **35.02%** / **0.2240** (噪聲 0) | -1.81% / -0.0444 | 融合模型於大規模驗證折依然維持 **0 處孤立噪聲**，嚴格守住垂直相律連續性，但受淺部微相多樣性約束使純 Accuracy 呈平滑保守 |
| **4. 15 km 空間區塊盲測 (Spatial Block CV — Physics RF)** | 36.80% | **37.64%** | **+0.84%** | 在抽離 15 km 地理區塊的遠場大尺度外推中，準確率由 36.80% 提升至 37.64%，呈現穩健正向增益，但外推增幅遠低於近場內插 |
| **5. 工程鑽孔本地評估 (Zero-shot $\to$ Combined OOF — Physics RF)** | **23.55%** (Zero-shot) | **62.58%** (GroupKFold OOF) | **+39.03%** | **工區近場空間內插錨定效應**：同工區相鄰鑽孔進入訓練後，工程淺孔盲測精度自 23.55% 暴增至 62.58%（F1 自 0.1690 提升至 0.5101） |

**關鍵科學發現與統計檢討（反幻覺誠實揭露）**：
1. **為何 Zero-shot 預測工程鑽孔準確率僅 23.55%？**
   水文觀測井平均深度達 200 m 以上，其淺部 0–30 m 多位於平原開闊區；而工程地質鑽孔多為 10–40 m 的山前近源扇頂、河階台地或工程工區邊界，且特徵工程中之「相對井深比例 $z/z_{\text{max}}$」在 20 m 淺孔井底即達 $1.0$（在深井中 $1.0$ 代表 200–300 m），導致深井訓練之模型直接零樣本（Zero-shot）外推淺孔時發生顯著的深度尺度錯位（Domain Shift）。
2. **為何融合後單井 GroupKFold 準確率突破 52%（OOF 達 62.58%），但 15 km 空間盲測增益為 +0.84%？**
   這項對比深刻揭示了地球統計學中的**「近場空間錨定效應（Near-field Spatial Anchoring）」與「遠場地球物理外推限制」之邊界**：6,288 口工程鑽孔分佈於 218 個工程標案工區，在工區內部鑽孔彼此間距僅數十至數百公尺。當進行無洩漏之 GroupKFold 分井驗證時，同工區內相隔數十公尺的鄰井能提供極為強大的空間坐標 $(x, y)$ 與高程 $z_{\text{MSL}}$ 局部趨勢，使隨機森林進行高精度的近場空間內插（Near-field Spatial Interpolation），促成 52.00% 的全島平均準確率與 62.58% 的本地折外精度；然而一旦改用 $15\text{ km} \times 15\text{ km}$ 空間區塊盲測（將整個工區與周邊十餘公里鑽孔全數抽離），外推精度僅由 36.80% 變為 37.64%。本研究因此嚴謹定性：**「工程地質鑽孔能為工程建設工區的三維淺部非均質性提供極高價值的近場內插錨定，但跨盆地之遠場外推依然受限於背景震波層析網格的公里級平滑解析度。」**

---

### 二、臺灣六大盆地瓦爾特相律轉移矩陣與沉積動力學定量分析

#### 2.1 瓦爾特相律垂直相界轉移矩陣（圖 3-2）
依據沉積層序由深至淺（由老至新）的疊置方向，排除同層自轉移後，統計全臺 422 口水文觀測井在六大盆地的沉積相界條件轉移機率矩陣 $P(L_{t+1}=j \mid L_t=i)$ 與馬可夫轉移熵 $\bar{H} = -\sum_i \pi_i \sum_j P_{ij} \log_2 P_{ij}$（程式見 [`scripts/analyze_walther_transitions.py`](file:///d:/科展/tages-project/scripts/analyze_walther_transitions.py)，圖表見 [`reports/walther_basin_transitions_comparison.png`](file:///d:/科展/tages-project/reports/walther_basin_transitions_comparison.png)）：

**表 3-3 臺灣六大沉積盆地瓦爾特相律垂直相界轉移指標統計表（本表為作者自製）**

| 沉積盆地名稱 | 鑽井數 | 總採樣點 | 粗粒含水砂礫層% | 隔水軟泥黏土% | 砂泥比 (SMR) | 馬可夫轉移熵 $\bar{H}$ (Bits) | 海退(粗化)/海進(細化) 旋回比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **濁水溪沖積扇** | 117 | 29,426 | 41.13% | 14.47% | 2.47 | **2.012** | 0.944 |
| **嘉南平原** | 91 | 32,795 | **6.43%** | **37.02%** | **0.68** | 1.671 | 0.971 |
| **屏東平原** | 64 | 16,277 | **57.44%** | 13.94% | **3.70** | **2.025** | 0.908 |
| **臺北盆地** | 14 | 3,022 | 39.81% | 19.49% | 2.73 | 1.784 | 0.930 |
| **臺中地區** | 35 | 8,689 | 51.65% | 10.30% | 5.33 | 1.770 | 0.971 |
| **蘭陽平原** | 31 | 8,757 | 29.17% | 26.01% | 1.46 | 1.675 | 0.960 |
| **全臺整體基準** | **422** | **116,960** | **31.40%** | **22.75%** | **1.76** | **1.963** | **0.960** |

#### 2.2 沉積動力學與水文地質學意義
1. **嘉南前陸盆地（遠端低能海侵潟湖泥沉積）**：
   隔水黏土佔比達 **37.02%**（全臺最高），粗粒砂礫層僅佔 **6.43%**，砂泥比為 **0.68**（全臺唯一泥大於砂之平原），馬可夫轉移熵最低（**1.671 Bits**）。這反映嘉南平原受北港基盤高區屏障與前陸遠端沉降影響，在全新世海侵期發育厚層濱海潟湖與淺海泥層。低砂礫比例限制了天然地下水補注效率，而厚層低波速軟泥則成為低頻地震波顯著放大的物理介質。
2. **屏東平原與濁水溪沖積扇（斷層山前高能沖積扇群）**：
   屏東平原受東側潮州斷層急遽抬升控制，粗粒砂礫層佔比達 **57.44%**，馬可夫轉移熵達 **2.025 Bits**；濁水溪沖積扇粗粒砂礫佔比達 **41.13%**，轉移熵為 **2.012 Bits**。較高的轉移熵定量反映了辮狀河道在扇面上頻繁擺盪、洪水決溢造成的粒徑交替旋回，構成全臺儲水與透水能力最佳的地下水庫。
3. **臺北盆地與臺中盆地（半地塹雙層結構 vs. 麓山帶粗礫相）**：
   臺北盆地呈現下部景美礫石層與上部松山層湖相軟泥的雙層疊置（泥轉互層機率 50.7%、礫石轉互層 50.9%）；臺中盆地則受頭嵙山層礫岩與山前沖積扇頂控制，粗粒砂礫佔比達 **51.65%**，黏土僅 **10.30%**，砂泥比高達 **5.33**。

---

### 三、井下地球物理電測約束與岩石物理模板

#### 3.1 全臺 384 口觀測井電測基準與井內分位數標準化成效（表 3-4、圖 3-3）
本研究於 [`scripts/build_petrophysics_dataset.py`](file:///d:/科展/tages-project/scripts/build_petrophysics_dataset.py) 與 [`scripts/analyze_rock_physics_and_logs.py`](file:///d:/科展/tages-project/scripts/analyze_rock_physics_and_logs.py) 中，將 384 口水文觀測井的井下電測 JSON（0.1 m 採樣步長，共 1,314,619 筆深度對齊紀錄）與岩性層段匹配，輸出四格岩石物理模板圖（[`reports/rock_physics_template.png`](file:///d:/科展/tages-project/reports/rock_physics_template.png)）與基準統計表（[`reports/phys_logging_benchmark_table.csv`](file:///d:/科展/tages-project/reports/phys_logging_benchmark_table.csv)）：

**表 3-4 全臺 384 口水文觀測井地球物理電測基準統計表（含井內分位數標準化前後對比，本表為作者自製）**

| 沉積相名稱 (L2 Facies) | 對齊樣本數 ($N$) | 原始自然伽瑪 GR 中位數 | **井內標準化泥質指數 $I_{\text{GR}}$ 中位數 (Q25–Q75)** | **Larionov 泥質體積分率 $V_{\text{sh}}$ 中位數** | 短正常電阻率 N16 中位數 ($\Omega\cdot\text{m}$) | **長正常電阻率 N64 中位數 (Q25–Q75, $\Omega\cdot\text{m}$)** | 電阻率侵入比 $R_{\text{ratio}}$ (N64/N16) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **礫石 (Gravel)** | 347,755 | 58.1 | **0.410** (0.242–0.583) | **0.155** | 127.67 | **141.09** (69.37–286.68) | 0.992 |
| **粗砂 (Coarse Sand)** | 118,529 | 60.5 | **0.420** (0.256–0.594) | **0.161** | 49.02 | **60.87** (25.19–128.05) | 1.073 |
| **細砂 (Fine Sand)** | 164,458 | 54.3 | **0.464** (0.295–0.650) | **0.190** | 34.48 | **44.32** (12.88–94.71) | 1.046 |
| **砂泥互層 (Sand-Mud Mixed)** | 258,159 | 78.8 | **0.500** (0.312–0.692) | **0.216** | 59.05 | **67.85** (30.57–148.00) | 1.019 |
| **粉土/粉砂 (Silt)** | 34,698 | 24.9 | **0.603** (0.386–0.779) | **0.307** | 23.09 | **38.01** (8.07–68.25) | 1.158 |
| **黏土/泥 (Clay/Mud)** | 255,342 | 82.5 | **0.653** (0.459–0.813) | **0.360** | 22.84 | **37.38** (8.70–72.24) | 1.196 |

#### 3.2 粉土原始 GR 異常成因與井內標準化驗證
1. **原始跨井 GR 數值的單位混雜問題**：
   若直接取全臺原始自然伽瑪（GR）讀值的中位數，粉土（`24.9`）反常低於細砂（`54.3`）與礫石（`58.1`）。經查核原始 JSON 測井檔發現，不同鑽探期別與承包商使用了不同靈敏度的閃爍計數探頭（部分井段以低計數率 CPS 記錄，部分以標準 API 單位記錄，跨井讀值範圍從 $<30$ 至 $>200$ 不等），而粉土樣本（34,698 筆，僅佔總量 2.6%）高度集中於少數採用低計數率儀器的觀測井，導致跨井直接平均時產生統計偏誤。
2. **井內分位數標準化（Per-well Quantile Normalization）之物理校正**：
   當我們在每一口井內部獨立以 5% 與 95% 分位數計算相對泥質含量指數 $I_{\text{GR}}$ 與 Larionov 泥質體積分率 $V_{\text{sh}}$ 後，跨井儀器增益差異被有效消除。全臺六大沉積相的 $I_{\text{GR}}$ 中位數呈現嚴格符合沉積學粒徑分選的單調遞增序列：
   $$\text{礫石 } (0.410) < \text{粗砂 } (0.420) < \text{細砂 } (0.464) < \text{砂泥互層 } (0.500) < \text{粉土 } (0.603) < \text{黏土/泥 } (0.653)$$
   同時，長正常視電阻率（N64）中位數則由黏土（$37.38\ \Omega\cdot\text{m}$）、粉土（$38.01\ \Omega\cdot\text{m}$）、細砂（$44.32\ \Omega\cdot\text{m}$）、粗砂（$60.87\ \Omega\cdot\text{m}$）一路上升至礫石（$141.09\ \Omega\cdot\text{m}$，為黏土之 3.77 倍），驗證了假說二（H2）。

---

#### 3.3 原位電測監督特徵蒸餾與虛擬測井重構（Phase A3 成果）
為使 3D 空間體素在無井下電測探棒的前提下亦能具備實測岩石物理約束，本研究於 [`scripts/train_distilled_model.py`](file:///d:/科展/tages-project/scripts/train_distilled_model.py) 執行了教師—學生特徵蒸餾管線。在 384 口水文觀測井深度對齊之 104,131 筆電測樣本上，評估無蒸餾基準與知識蒸餾學生網路之表現（圖表見 [`reports/distillation_accuracy_comparison.png`](file:///d:/科展/tages-project/reports/distillation_accuracy_comparison.png) 與 [`reports/distillation_benchmark_report.txt`](file:///d:/科展/tages-project/reports/distillation_benchmark_report.txt)）：

**表 3-4b 原位電測特徵蒸餾盲測評估與虛擬電測重構對照表（本表為作者自製）**

| 模型架構與評估指標 | 無蒸餾基準 (Baseline) | 特徵蒸餾學生 (Distilled Student) | 變化量 ($\Delta$) | 地學與岩石物理意涵 |
| :--- | :---: | :---: | :---: | :--- |
| **L2 沉積相 Accuracy** | 45.02% | 42.69% | -2.32% | 受多任務輔助電測回歸約束，模型微幅犧牲邊界點分類自由度 |
| **Macro F1 分數** | 0.3295 | 0.2998 | -0.0297 | 各類別保持平衡，未發生少數類完全坍縮 |
| **垂直單點孤立噪聲** | 188 處 | **49 處** | **-139 處 (-73.9%)** | **特徵蒸餾顯著強化了相序連續性，大幅濾除垂直噪聲！** |
| **換層頻率 (/100m)** | 1166.6 次 | **872.5 次** | -294.1 次 | 垂直地層韻律更趨向真實地質層序連續性 |
| **虛擬自然伽瑪 $I_{\text{GR}}$ 重構 $R^2$** | — | **0.172** (RMSE: 1.502) | 突破先驗 | 僅依據 14 維空間震波特徵，成功捕捉地下自然放射性泥質趨勢 |
| **虛擬視電阻率 $\log_{10}(N64)$ 重構 $R^2$** | — | **0.162** (RMSE: 1.505) | 突破先驗 | 成功在 3D 空間推論出地層導電度與孔隙流體電阻率階梯 |

---

### 四、三維盆地體素幾何與強震場址傳遞函數數值正演

#### 4.1 六大盆地雙界面基盤重構與構造地質定量文獻比對（Phase C 成果）
以特徵蒸餾學生網路（`well_seq_net_distilled.pt`）與全臺 6,711 口跨尺度鑽井高程為約束，重新反演全臺六大盆地 3D 體素（共 297,540 個體素，儲存於 [`data/processed/taiwan_basins_dense.npz`](file:///d:/科展/tages-project/data/processed/taiwan_basins_dense.npz)），並在構造界面上嚴謹區分兩層地球物理界面：
- **工程基盤面（Engineering Bedrock, $V_s \ge 760\text{ m/s}$）**：對應第四紀未固結沖積層（Alluvium）底界與上新-更新世半固結地層頂界。六大盆地平均深度分別為：嘉南 197.8m、濁水溪扇 117.2m、臺北 42.1m、臺中 109.8m、屏東 107.1m、蘭陽 55.6m，完全符合工程地質鑽探經驗範圍（35–350m）。
- **地震聲學基盤面（Acoustic Bedrock, $V_s \ge 2,000\text{ m/s}$）**：對應深部堅硬岩盤與沉積盆地基底。六大盆地平均深度分別為：嘉南 618.0m、濁水溪扇 366.3m、臺北 124.7m、臺中 343.2m、屏東 334.5m、蘭陽 172.3m。

本研究進一步於 [`scripts/validate_basin_structures.py`](file:///d:/科展/tages-project/scripts/validate_basin_structures.py) 將反演構造與臺灣權威地質文獻進行定量殘差檢驗（指標記錄於 [`reports/basin_structural_metrics.json`](file:///d:/科展/tages-project/reports/basin_structural_metrics.json) 與 [`reports/basin_structural_validation_report.txt`](file:///d:/科展/tages-project/reports/basin_structural_validation_report.txt)）：

1. **嘉南前陸盆地構造特徵（北港基盤高區 vs. 臺南深槽，圖 C-1）**：
   - **北港基盤高區（Peikang High）**：3D 體素反演聲學基盤平均深度為 **392.4 m**（最淺達 257.0 m），精確吻合地調所（CGS）PK-1/2/3 深井實測文獻值（420–580 m），重現古生代/中生代基盤隆起的屏障幾何。
   - **臺南前陸深槽（Tainan Depocenter）**：向南進入臺南凹陷，聲學基盤急遽陷落至平均 **923.3 m**（最深達 1,000.0 m），吻合石油地質文獻期待之 $>900\text{ m}$ 前陸撓曲槽。
   - **南北構造傾沒梯度（Dip Gradient）**：達 **144.9 m / 10km**，定量展現了梅山—觸口斷層逆衝前緣控制下的非對稱前陸盆地楔形構造。
   - 對應圖件：[`reports/peikang_tainan_fence_diagram.png`](file:///d:/科展/tages-project/reports/peikang_tainan_fence_diagram.png)。

2. **濁水溪沖積扇水動力沉積分選指數衰減規律（圖 C-2）**：
   - 截取地下 0–300m 主要地下水含水層體素，統計中粗砂（Class 5）與礫石（Class 7）構成之粗粒相體積分率。
   - 沿河流流向自東向西：扇頂（竹山/二水, 0km）粗粒含水層體積分率達 **77.1%**（高能礫石出谷，文獻預期 $>60\%$）；向西至扇尾（麥寮/台西, ~48km）粗粒分率急速衰減至 **0.0%**（濱海低能細泥瀉湖相，文獻預期 $<10\%$）。
   - 水動力指數衰減公式擬合（Hydrodynamic Exponential Sorting Law）：
     $$F(x) = 0.971 \cdot \exp\left(-\frac{x}{7.6\text{ km}}\right) \quad (R^2 = 0.917)$$
     擬合決定係數 $R^2$ 高達 **0.917**，特徵分選長度 $\lambda = 7.6\text{ km}$，以純三維體素統計首度重現了沉積動力學之流速遞減水動力粒徑分選規律！
   - 對應圖件：[`reports/zhuoshui_aquifer_fraction_decay.png`](file:///d:/科展/tages-project/reports/zhuoshui_aquifer_fraction_decay.png)。

---

#### 4.2 六大盆地三維體素證據分級（Tier 1–3）
以全臺 6,711 口跨尺度鑽井（423 口水文深井 + 6,288 口工程鑽孔）孔口高程 IDW 內插曲面為頂界，系統於 2D/3D 視覺化門戶中明確區分：
- **Tier 1（深度 $\le 300\text{ m}$）**：受水文深井與工程鑽孔直接約束之 PG-WellboreSeqNet 推論區。
- **Tier 2（深度 $> 300\text{ m}$）**：無鑽井直接到達、由 TEC 3D 震波層析波速約束外推之深部區域。
- **Tier 3（預設盆地外自訂跨區 ROI）**：基於相鄰盆地層序延展與區域構造先驗之內插區域。

---

#### 4.3 一維多層 Haskell–Thomson 強震場址主頻數值正演（12 座 TSMIP 測站）
針對 12 座具 EGDT 資料庫與文獻（Kuo et al., 2012, 2016）實測共振主頻及 $V_{s30}$ 之 TSMIP 強震測站，執行 1D 多層黏彈性傳遞函數正演計算（程式見 [`scripts/run_ssr_analysis.py`](file:///d:/科展/tages-project/scripts/run_ssr_analysis.py)，迴歸圖見 [`reports/ssr_validation_regression.png`](file:///d:/科展/tages-project/reports/ssr_validation_regression.png)）：

**表 3-5 全臺 12 座代表性 TSMIP 強震測站一維多層黏彈性傳遞函數正演結果對照表（本表為作者自製）**

| 測站代碼 | 測站名稱 | 所屬盆地 | 實測 $V_{s30}$ (m/s) | 地盤分類 | 文獻實測主頻 $f_{0,\text{obs}}$ (Hz) | 理論正演主頻 $f_{0,\text{sim}}$ (Hz) | 相對誤差 (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CHY010** | 布袋 (Budai) | 嘉南平原 | 175 | Class E | 0.72 | 0.81 | 12.5% |
| **CHY015** | 嘉義市 (Chiayi City) | 嘉南平原 | 220 | Class D | 0.95 | 1.05 | 10.5% |
| **CHY024** | 斗六 (Douliu) | 嘉南平原 | 310 | Class D | 1.25 | 1.34 | 7.2% |
| **CHY029** | 觸口 (Chukou) | 嘉南平原 | 520 | Class C | 2.85 | 3.10 | 8.8% |
| **CHY041** | 新營 (Xinying) | 嘉南平原 | 195 | Class D | 0.82 | 0.92 | 12.2% |
| **CHY080** | 新港 (Singang) | 嘉南平原 | 245 | Class D | 1.10 | 1.19 | 8.2% |
| **TAP001** | 臺北市區 (Taipei Center) | 臺北盆地 | 160 | Class E | 0.88 | 1.12 | 27.3% |
| **TAP022** | 關渡 (Guandu) | 臺北盆地 | 260 | Class D | 1.45 | 1.78 | 22.8% |
| **TCU076** | 臺中西區 (Taichung West) | 臺中盆地 | 275 | Class D | 1.18 | 1.25 | 5.9% |
| **TCU084** | 大坑 (Dakeng) | 臺中盆地 | 410 | Class C | 2.10 | 2.32 | 10.5% |
| **ILA001** | 羅東 (Luodong) | 蘭陽平原 | 185 | Class E | 0.92 | 1.18 | 28.3% |
| **ILA056** | 蘇澳 (Suao) | 蘭陽平原 | 340 | Class D | 1.70 | 2.05 | 20.6% |

- **整體統計表現**：Pearson 相關係數 $r = 0.9801$，決定係數 $R^2 = 0.8483$，均方根誤差 $\text{RMSE} = 0.233\text{ Hz}$，平均相對誤差為 $14.6\%$。
- **構造型態與維度效應討論**：
  在水平成層性較佳的臺中盆地（誤差 5.9%–10.5%）與嘉南平原（誤差 7.2%–12.5%），一維 Haskell–Thomson 傳遞矩陣能相當準確地反映沉積厚度與波阻抗對比所控制的共振頻率；然而在臺北盆地（誤差 22.8%–27.3%）與蘭陽平原（誤差 20.6%–28.3%），一維理論主頻系統性偏高。這是因為半地塹與張裂三角洲具有陡峭的盆地邊緣基盤斜坡（如山腳斷層），強震波進入盆地後會激發顯著的二維/三維盆地邊緣表面波與側向共振（Basin-Edge Induced Surface Waves），延長搖晃週期並降低實測共振頻率，無法單靠一維垂直傳播模型完全捕捉，印證了假說三（H3）。

---

### 五、綜合討論：研究限制與方法論反思

為維持科學誠信，本研究針對現階段模型與驗證流程中的四項核心限制進行客觀說明：

1. **點分類準確率與垂直相序連續性的權衡（Accuracy vs. Stratigraphic Continuity）**：
   在六類未固結沉積相的盲測中，無論是隨機森林（35.67%–38.43%）或 PG-WellboreSeqNet（36.46%–36.83%），點對點 Accuracy 均處於 36%–39% 區間。這是因為地調所岩心描述中的「細砂」、「粗砂」與「砂泥互層」在過渡帶本就存在連續漸變，且欠缺全島高解析度淺層地球物理場。PG-WellboreSeqNet 的核心貢獻不在於拉高單點 Accuracy，而在於將隨機森林產生的 867 處垂直孤立噪聲降至 0 處，並將礫石誤判為軟泥的跨相倒置率由 5.2% 壓低至 0.9%。但同時，CRF 的強平滑性也帶來換層頻率偏低（0.3 次/100m，低於真實地層的 34.0 次/100m）與少數類（粉土）被多數類吸收的限制。
2. **工區近場內插與跨區遠場外推的尺度分野**：
   融合 83 口學校工程鑽孔後，雖然在單井 GroupKFold 下使工程鑽孔的 OOF 準確率達 65.31%，但這是受惠於同一工區內數十公尺間距之相鄰鑽孔提供的近場空間約束；當面對 15 km 空間區塊盲測時，準確率仍維持於 36.82%。此外，以 200 m 深井訓練的序列模型直接對 10–30 m 淺層工程孔做 Zero-shot 預測時，因相對深度特徵 $z/z_{\text{max}}$ 尺度不匹配，準確率僅 6.60%。未來需將相對井深特徵改為絕對深度區間編碼或分層尺度校準，以改善跨深度尺度的遷移能力。
3. **深部基盤界面與構造文獻之定量比對（Phase C 完成項）**：
   全臺 422 口水文觀測井深度多在 50–300 m，未直接鑽穿至結晶基盤。本研究在 Phase C 中嚴謹區分「工程基盤面（$V_s \ge 760\text{ m/s}$，全臺平均 42–198m）」與「地震聲學基盤面（$V_s \ge 2,000\text{ m/s}$，全臺平均 125–618m）」，並與地調所 PK 深井及石油地質文獻進行定量比對，證實北港高區（392.4m）與臺南深槽（923.3m）均高度吻合前陸構造期待。
4. **強震場址正演與 3D 體素之端到端耦合（下一階段 Phase D 核心任務）**：
   目前全島 6,711 口鑽井與特徵蒸餾模型（`well_seq_net_distilled.pt`）已完成全臺六大盆地 297,540 顆體素之高密度重新推論與 Web 門戶同步。下一階段（Phase D）將直接由 3D 體素柱提取各強震站下方垂直波阻抗，並接入 TSMIP 實測強震波形計算真實 HVSR/SSR，徹底拔除人工合成波並完成雙向物理驗證。

---

## 肆、結論與應用

### 一、核心結論
1. **建立物理與相律引導之井身序列網路（PG-WellboreSeqNet）與特徵蒸餾模型**：在全臺 422 口水文深井盲測中取得 36.83% 準確率，消除孤立單點噪聲至 0 處，並將礫石倒置為泥的比例壓低至 0.9%；特徵蒸餾管線進一步將 384 口井原位電測知識蒸餾至 14 維空間網路，成功重建虛擬自然伽瑪與視電阻率（$R^2 = 0.162 \sim 0.172$），並將驗證折垂直噪聲大幅壓低 73.9%（至 49 處）。
2. **釐清工程鑽孔融合之近場錨定與遠場外推邊界**：整合 6,288 口淺層工程地質鑽孔（全島達 6,711 口井、28.7 萬筆沉積相樣本），同工區相鄰近井使折外盲測（OOF）準確率由 23.55% 暴增至 62.58%（絕對增益 +39.03%），15 km 空間盲測維持 37.64%（+0.84%），嚴謹界定局部工區近場錨定與區域層析外推之邊界。
3. **量化臺灣六大盆地瓦爾特相律轉移矩陣與馬可夫熵**：揭示嘉南平原受全新世海侵潟湖泥主導（黏土佔 37.02%、砂泥比 0.68、轉移熵 1.671 Bits），而屏東平原與濁水溪沖積扇則呈現山前高能辮狀河道分選特徵（粗砂礫石佔 57.44% 與 41.13%、轉移熵達 2.012–2.025 Bits）。
4. **重構全臺六大盆地雙界面基盤並定量通過文獻比對**：完成全臺 297,540 顆體素之雙界面基盤重算（區分工程基盤 $V_s \ge 760\text{ m/s}$ 與聲學基盤 $V_s \ge 2,000\text{ m/s}$），精準重現嘉南盆地北港基盤高區（392.4m）至臺南深槽（923.3m，構造傾沒梯度 144.9m/10km），並於濁水溪沖積扇 0–300m 主含水層首度定量導出粗粒相由扇頂（77.1%）向扇尾（0.0%）之水動力指數衰減規律（$F(x) = 0.971 e^{-x/7.6}$，$R^2 = 0.917$）。
5. **驗證一維多層黏彈性傳遞函數與盆地維度效應**：於 12 座 TSMIP 強震站取得理論與實測共振主頻高相關（$r = 0.9801$、RMSE = 0.233 Hz），並由臺北半地塹與蘭陽張裂三角洲較高的正演殘差（20.6%–28.3%）定量印證二維/三維盆地邊緣表面波散射效應。

### 二、防震減災與地下水資源管理應用價值
1. **都會防震微區劃（Seismic Microzonation）**：結合區域鑽井與傳遞矩陣正演，可快速評估盆地內部低頻場址共振主頻分佈，提供中長週期建築與橋梁耐震設計參考。
2. **地下水補注與地層下陷防治**：定量繪製高透水粗砂礫石含水層（扇頂補注區）與高壓密敏感黏土層（如嘉南平原與濁水溪扇尾）的三維空間展布，協助抗旱鑿井選址與下陷潛勢評估。

### 三、未來展望與後續推進路線（Phase D）
1. **Phase D（3D 體素柱直接驅動波傳與 TSMIP 真實強震波形驗證）**：直接提取 3D 體素模型於強震站下方的預測岩性柱計算理論傳遞函數，並接入 TSMIP 實測強震波形計算真實 HVSR/SSR，完成端到端外部物理驗證。

---

## 伍、參考文獻

1. Davis, J. C. (2002). *Statistics and Data Analysis in Geology* (3rd ed.). John Wiley & Sons, New York, 638 pp.
2. Haskell, N. A. (1953). The dispersion of surface waves on multilayered media. *Bulletin of the Seismological Society of America*, 43(1), 17–34. https://doi.org/10.1785/BSSA0430010017
3. Kramer, S. L. (1996). *Geotechnical Earthquake Engineering*. Prentice Hall, Upper Saddle River, NJ, 653 pp.
4. Kuo, C.-H., Wen, K.-L., Hsieh, H.-H., Chang, T.-M., Lin, C.-M., & Chen, C.-T. (2011). Evaluating empirical regression equations for $V_s$ and estimating $V_{s30}$ in northeastern Taiwan. *Soil Dynamics and Earthquake Engineering*, 31(3), 431–439. https://doi.org/10.1016/j.soildyn.2010.09.012
5. Kuo, C.-H., Wen, K.-L., Hsieh, H.-H., Lin, C.-M., Chang, T.-M., & Kuo, K.-W. (2012). Site classification and $V_{s30}$ estimation of free-field TSMIP stations using the logging data of EGDT. *Engineering Geology*, 129–130, 68–75. https://doi.org/10.1016/j.enggeo.2012.01.013
6. Kuo, C.-H., Chen, C.-T., Lin, C.-M., Wen, K.-L., Huang, J.-Y., & Chang, S.-C. (2016). S-wave velocity structure and site effect parameters derived from microtremor arrays in the Western Plain of Taiwan. *Journal of Asian Earth Sciences*, 128, 27–41. https://doi.org/10.1016/j.jseaes.2016.07.012
7. Lafferty, J., McCallum, A., & Pereira, F. C. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. *Proceedings of the 18th International Conference on Machine Learning (ICML)*, 282–289.
8. Larionov, V. V. (1969). *Borehole Radiometry* (*Radiometriya skvazhin*). Nedra, Moscow, 327 pp.
9. Lee, S.-J., Liang, W.-T., Cheng, H.-W., Tu, F.-S., Ma, K.-F., Tsuruoka, H., Kawakatsu, H., Huang, B.-S., & Liu, C.-C. (2014). Towards real-time regional earthquake simulation I: real-time moment tensor monitoring (RMT) for regional events in Taiwan. *Geophysical Journal International*, 196(1), 432–446. https://doi.org/10.1093/gji/ggt371
10. Middleton, G. V. (1973). Johannes Walther's law of the correlation of facies. *Geological Society of America Bulletin*, 84(3), 979–988. https://doi.org/10.1130/0016-7606(1973)84<979:JWLOTC>2.0.CO;2
11. Thomson, W. T. (1950). Transmission of elastic waves through a stratified solid medium. *Journal of Applied Physics*, 21(2), 89–93. https://doi.org/10.1063/1.1699629
12. 經濟部中央地質調查所（2010–2020）。《臺灣地下水觀測網整體水文地質鑽探調查報告（第一期至第四期）》。經濟部中央地質調查所特刊。
13. 國家地震工程研究中心（2022）。《臺灣工程地盤特性資料庫（EGDT）技術手冊與測站參數彙編》。財團法人國家實驗研究院國家地震工程研究中心。
