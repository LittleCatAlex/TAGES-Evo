# -*- coding: utf-8 -*-
"""
產出科展作品說明書風格之「研究流程圖」
對標全國/國際科展作品說明書標準版型與使用者參考圖樣式
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# 設置微軟正黑體以正確顯示繁體中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_flowchart(output_path="c:/Users/alex2/科展/TAGES-Evo/reports/research_flowchart.png"):
    fig = plt.figure(figsize=(15.5, 11.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # 顏色調色盤 (參考科展經典配色：藍、綠、青灰、深藍)
    c_banner = '#2B70C9'       # 最上方主標題藍
    c_sub_bg = '#EBF3FB'       # 上方資料整合淺藍
    c_border_blue = '#1E5AA8'
    c_green_box = '#E8F5E9'    # 左側地質序列綠色背景
    c_green_border = '#2E7D32'
    c_green_subcard = '#FFFFFF'
    c_blue_box = '#E0F7FA'     # 中間電測蒸餾淺藍背景
    c_blue_border = '#00838F'
    c_blue_subcard = '#FFFFFF'
    c_purple_box = '#EDE7F6'   # 右側波傳正演淺紫藍背景
    c_purple_border = '#4527A0'
    c_purple_subcard = '#FFFFFF'

    # 1. 頂部總標題 Banner (深藍色圓角矩形)
    banner = FancyBboxPatch((5, 91.5), 90, 5.8, boxstyle="round,pad=0.5,rounding_size=1.5",
                            facecolor=c_banner, edgecolor=c_border_blue, linewidth=2, zorder=2)
    ax.add_patch(banner)
    ax.text(50, 94.4, "結合物理先驗序列深度學習之三維地質重構與臺灣主要沉積盆地強震場址效應預測",
            color='white', fontsize=14.5, fontweight='bold', ha='center', va='center', zorder=3)

    # 下拉箭頭 1
    ax.annotate('', xy=(50, 89.2), xytext=(50, 91.5),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)

    # 2. 第二層：多源地學資料與先驗特徵工程
    data_box = FancyBboxPatch((7, 76.5), 86, 12.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                              facecolor=c_sub_bg, edgecolor=c_border_blue, linewidth=1.5, zorder=2)
    ax.add_patch(data_box)
    ax.text(50, 86.8, "【多源異質地球物理資料庫整合與物理特徵工程】",
            color='#0C356A', fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)

    # 4個資料源膠囊
    d_items = [
        ("水文地質深層觀測井", "422 口深井 / 11.7 萬筆採樣\n公尺級沉積相基準與相界統計"),
        ("GSMMA 公開工程鑽孔", "34,853 口孔 / 105.6 萬樣點\n淺表 0–40m 都會工區高密錨定"),
        ("井下原位地球物理電測", "384 口井 / 131.5 萬筆數據\n自然伽瑪 GR 與視電阻率 N64"),
        ("TEC LEE2025 層析模型", "全島 2,600 節點三維網格\n背景波速 Vp, Vs 與動態泊松比 ν")
    ]
    for i, (head, sub) in enumerate(d_items):
        bx = 9 + i * 20.6
        subb = FancyBboxPatch((bx, 79.8), 19.2, 5.6, boxstyle="round,pad=0.3,rounding_size=0.8",
                              facecolor='white', edgecolor='#5C8374', linewidth=1.2, zorder=3)
        ax.add_patch(subb)
        ax.text(bx + 9.6, 83.6, head, color='#040D12', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(bx + 9.6, 81.3, sub, color='#183D3D', fontsize=7.2, ha='center', va='center', zorder=4, linespacing=1.2)

    # 特徵工程橫條
    feat_box = FancyBboxPatch((9, 77.2), 82, 1.9, boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor='#DDF2FD', edgecolor='#427D9D', linewidth=1, zorder=3)
    ax.add_patch(feat_box)
    ax.text(50, 78.15, "關鍵特徵先驗：固定對數深度 z_log ｜ 多尺度空間傅立葉編碼 ｜ 零洩漏雙尺度近場先驗 ｜ 瓦爾特相律轉移矩陣",
            color='#164863', fontsize=8.2, ha='center', va='center', zorder=4)

    # 下拉分流箭頭
    ax.annotate('', xy=(50, 73.8), xytext=(50, 76.5),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)
    # 水平分流線
    ax.plot([22, 78], [73.8, 73.8], color='#4A5568', linewidth=2, zorder=4)
    ax.annotate('', xy=(22, 71.8), xytext=(22, 73.8),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)
    ax.annotate('', xy=(50, 71.8), xytext=(50, 73.8),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)
    ax.annotate('', xy=(78, 71.8), xytext=(78, 73.8),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)

    # 3. 三大核心支柱 (Column Containers)
    # ---------------- 支柱 1: 地質序列建模 (左, 綠色) ----------------
    col1 = FancyBboxPatch((5.5, 17), 28, 54.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                          facecolor=c_green_box, edgecolor=c_green_border, linewidth=2, zorder=2)
    ax.add_patch(col1)
    # 支柱 1 標題
    h1 = FancyBboxPatch((7, 67.2), 25, 3.8, boxstyle="round,pad=0.3,rounding_size=0.8",
                        facecolor=c_green_border, edgecolor='#1B5E20', linewidth=1.5, zorder=3)
    ax.add_patch(h1)
    ax.text(19.5, 69.1, "【相律約束地質序列建模】\nPG-WellboreSeqNet", color='white',
            fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=4, linespacing=1.2)

    steps_col1 = [
        ("多尺度 1D-ResNet 卷積", "並聯 3m/5m/9m 卷積核 + SE 通道注意力\n同質長度分桶消除零填充偏置"),
        ("雙向長短期記憶網路", "2 層 BiLSTM (隱藏層 H=96, 拼接 192 維)\n同時捕捉由深至淺與覆壓疊置關係"),
        ("少數類對數先驗補償", "Logit Adjustment 先驗補償閘門\n粉土召回率由 0.0% 恢復至 5.12%"),
        ("列中心化相律隨機場", "瓦爾特相律轉移矩陣約束之線性鏈 CRF\n對角線降權抑制過度單相平滑"),
        ("單點尖峰濾除與驗證", "全島 28,969 口井 5 折盲測達 55.66%\n垂直孤立跳變噪聲：4,294 處 -> 0 處")
    ]
    for idx, (stitle, sdesc) in enumerate(steps_col1):
        sy = 57.5 - idx * 9.6
        sp = FancyBboxPatch((7.5, sy), 24, 7.6, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor=c_green_subcard, edgecolor='#81C784', linewidth=1.2, zorder=3)
        ax.add_patch(sp)
        ax.text(19.5, sy + 5.7, f"步驟 1.{idx+1}：{stitle}", color='#1B5E20', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(19.5, sy + 2.7, sdesc, color='#2E7D32', fontsize=7.2, ha='center', va='center', zorder=4, linespacing=1.2)
        if idx < 4:
            ax.annotate('', xy=(19.5, sy - 1.8), xytext=(19.5, sy),
                        arrowprops=dict(facecolor='#4CAF50', edgecolor='#4CAF50', width=1.5, headwidth=5, headlength=6), zorder=4)

    # ---------------- 支柱 2: 電測蒸餾與 3D 體素重構 (中, 藍色) ----------------
    col2 = FancyBboxPatch((36, 17), 28, 54.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                          facecolor=c_blue_box, edgecolor=c_blue_border, linewidth=2, zorder=2)
    ax.add_patch(col2)
    # 支柱 2 標題
    h2 = FancyBboxPatch((37.5, 67.2), 25, 3.8, boxstyle="round,pad=0.3,rounding_size=0.8",
                        facecolor=c_blue_border, edgecolor='#006064', linewidth=1.5, zorder=3)
    ax.add_patch(h2)
    ax.text(50, 69.1, "【原位電測蒸餾與 3D 重構】\nPetrophysics & 3D Voxels", color='white',
            fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=4, linespacing=1.2)

    steps_col2 = [
        ("教師網路原位電測預訓練", "17 維輸入 (14 維空間震波 + 3 維電測)\n384 口觀測井、10.4 萬筆樣本預訓練"),
        ("學生網路特徵知識蒸餾", "14 維無測井輸入 (學生網路 DistilledNet)\nKL 散度與溫度 T=2.5 遷移岩石物理特徵"),
        ("虛擬電測曲線物理重構", "泥質指數 I_GR 重構 R²=0.182 (RMSE=0.258)\n長正常電阻率 N64 重構 R²=0.121"),
        ("雙界面基盤構造先驗", "工程基盤面 (Vs >= 760 m/s，淺層共振界線)\n聲學基盤面 (Vs >= 2,000 m/s，深部固結基底)"),
        ("六大沉積盆地三維體素化", "反演全臺 297,540 顆三維空間體素\n重現濁水溪沖積扇水動力衰減 (R²=0.936)")
    ]
    for idx, (stitle, sdesc) in enumerate(steps_col2):
        sy = 57.5 - idx * 9.6
        sp = FancyBboxPatch((38, sy), 24, 7.6, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor=c_blue_subcard, edgecolor='#4DD0E1', linewidth=1.2, zorder=3)
        ax.add_patch(sp)
        ax.text(50, sy + 5.7, f"步驟 2.{idx+1}：{stitle}", color='#006064', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(50, sy + 2.7, sdesc, color='#00838F', fontsize=7.2, ha='center', va='center', zorder=4, linespacing=1.2)
        if idx < 4:
            ax.annotate('', xy=(50, sy - 1.8), xytext=(50, sy),
                        arrowprops=dict(facecolor='#00ACC1', edgecolor='#00ACC1', width=1.5, headwidth=5, headlength=6), zorder=4)

    # ---------------- 支柱 3: 強震場址波傳正演 (右, 紫色) ----------------
    col3 = FancyBboxPatch((66.5, 17), 28, 54.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                          facecolor=c_purple_box, edgecolor=c_purple_border, linewidth=2, zorder=2)
    ax.add_patch(col3)
    # 支柱 3 標題
    h3 = FancyBboxPatch((68, 67.2), 25, 3.8, boxstyle="round,pad=0.3,rounding_size=0.8",
                        facecolor=c_purple_border, edgecolor='#311B92', linewidth=1.5, zorder=3)
    ax.add_patch(h3)
    ax.text(80.5, 69.1, "【強震場址波傳正演與驗證】\nHaskell–Thomson Closed Loop", color='white',
            fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=4, linespacing=1.2)

    steps_col3 = [
        ("測站正下方體素地層柱提取", "全臺 32 座 NCREE EGDT 強震測站\n直接自 3D 體素切出岩性序列與基盤深度"),
        ("逐層壓密波速與阻尼映射", "全臺統一經驗參數：有效深度壓密指數 0.25\n岩相密度 ρ、波速 Vs、衰減品質因子 Qs"),
        ("1D 黏彈性傳遞矩陣求解", "Haskell–Thomson 多層 SH 波傳遞矩陣\n求解地表相對於基盤之連續放大譜 |H(f)|"),
        ("32 站共振主頻閉環盲測", "預測主頻 vs 基準觀測：r=0.8224, R²=0.5831\nRMSE=0.316 Hz (較四分之一波長誤差降 76.5%)"),
        ("盆地邊緣幾何效應空間回歸", "平原中心 (>= 6km) 正演偏差僅 +0.002 Hz\n定量證實邊緣 6km 內 2D/3D 表面波散射")
    ]
    for idx, (stitle, sdesc) in enumerate(steps_col3):
        sy = 57.5 - idx * 9.6
        sp = FancyBboxPatch((68.5, sy), 24, 7.6, boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor=c_purple_subcard, edgecolor='#9FA8DA', linewidth=1.2, zorder=3)
        ax.add_patch(sp)
        ax.text(80.5, sy + 5.7, f"步驟 3.{idx+1}：{stitle}", color='#283593', fontsize=9, fontweight='bold', ha='center', va='center', zorder=4)
        ax.text(80.5, sy + 2.7, sdesc, color='#3949AB', fontsize=7.2, ha='center', va='center', zorder=4, linespacing=1.2)
        if idx < 4:
            ax.annotate('', xy=(80.5, sy - 1.8), xytext=(80.5, sy),
                        arrowprops=dict(facecolor='#5C6BC0', edgecolor='#5C6BC0', width=1.5, headwidth=5, headlength=6), zorder=4)

    # 橫向資料流向箭頭：支柱1 -> 支柱2 -> 支柱3
    ax.annotate('', xy=(36, 21), xytext=(33.5, 21),
                arrowprops=dict(facecolor='#1E5AA8', edgecolor='#1E5AA8', width=2, headwidth=6, headlength=7), zorder=5)
    ax.annotate('', xy=(66.5, 21), xytext=(64, 21),
                arrowprops=dict(facecolor='#1E5AA8', edgecolor='#1E5AA8', width=2, headwidth=6, headlength=7), zorder=5)

    # 4. 底部綜合成果與工程決策 Banner (連接三支柱輸出)
    ax.plot([19.5, 80.5], [14.8, 14.8], color='#4A5568', linewidth=2, zorder=4)
    ax.plot([19.5, 19.5], [17, 14.8], color='#4A5568', linewidth=2, zorder=4)
    ax.plot([50, 50], [17, 14.8], color='#4A5568', linewidth=2, zorder=4)
    ax.plot([80.5, 80.5], [17, 14.8], color='#4A5568', linewidth=2, zorder=4)
    ax.annotate('', xy=(50, 12.8), xytext=(50, 14.8),
                arrowprops=dict(facecolor='#4A5568', edgecolor='#4A5568', width=2, headwidth=7, headlength=8), zorder=4)

    bottom_box = FancyBboxPatch((5, 4.8), 90, 7.8, boxstyle="round,pad=0.5,rounding_size=1.5",
                                facecolor='#1B5E20', edgecolor='#0D3810', linewidth=2, zorder=2)
    ax.add_patch(bottom_box)
    ax.text(50, 10.3, "【科研成果整合、工程防震應用與耐震規範改革實踐】",
            color='#FFD54F', fontsize=12, fontweight='bold', ha='center', va='center', zorder=3)
    b_desc = (
        "① 嘉南平原 1,296 點空間微區劃 (長週期共振主頻與場址放大倍率分佈圖)   "
        "② 2016 美濃地震真實波形驗證與深部基盤 Z_1.0 控制規律實證 (r=0.7478)\n"
        "③ 量化現行耐震規範盲點：西部三大深盆地長週期 (T=1.5s) 平均低估 45.2% (最大 69.5%)   "
        "④ 3D 數位分身門戶全域即時場址效應解算器 (純 JS 即時演算)"
    )
    ax.text(50, 7.3, b_desc, color='white', fontsize=8.6, ha='center', va='center', zorder=3, linespacing=1.35)

    # 5. 說明文字與圖題 (遵循科展規範)
    plt.figtext(0.5, 0.015, "圖 1. 研究流程與技術方法架構圖（圖片來源：作者自行繪製）",
                ha='center', fontsize=12, fontweight='bold', color='#1A202C')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Flowchart successfully created at: {output_path}")

if __name__ == '__main__':
    create_flowchart()
