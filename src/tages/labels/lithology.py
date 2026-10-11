"""岩性描述（繁中）→ 地層岩性分類 (支援 L1 粗類 7 類 與 L2 細分 10 類)。

L1 類別 (基礎 7 類):
  0 Unknown | 1 Fill | 2 ClaySilt | 3 Sand | 4 Gravel | 5 Interbedded | 6 SoftRock | 7 HardRock

L2 類別 (地質學與工程分類細化 10 類):
  0 Unknown     - 資料空缺 / 套管
  1 Fill        - 表土 / 回填土 / 人工填土
  2 ClayMud     - 黏土 / 泥 (高塑性細粒沉積物)
  3 Silt        - 粉土 / 粉砂 (低塑性細粒沉積物)
  4 FineSand    - 細砂 / 極細砂 (沖積平原主沉積砂相)
  5 CoarseSand  - 中砂 / 粗砂 / 礫質砂 (高能河道砂相)
  6 SandMudMixed- 砂泥互層 / 泥砂過渡相
  7 Gravel      - 礫石 / 卵石 / 巨礫 (沖積扇與河道滯留相)
  8 Mudstone    - 泥岩 / 頁岩 (第三紀固結/半固結軟岩基盤)
  9 Sandstone   - 砂岩 / 礫岩 (第三紀固結硬砂岩基盤)
  10 Basement   - 結晶硬基盤 / 火成岩 / 變質岩 (前第三紀基底)
"""
from __future__ import annotations
import re

L1_NAMES = {
    0: "Unknown", 1: "Fill", 2: "ClaySilt", 3: "Sand",
    4: "Gravel", 5: "Interbedded", 6: "SoftRock", 7: "HardRock",
}

L2_NAMES = {
    0: "Unknown",
    1: "Fill",
    2: "ClayMud",
    3: "Silt",
    4: "FineSand",
    5: "CoarseSand",
    6: "SandMudMixed",
    7: "Gravel",
    8: "Mudstone",
    9: "Sandstone",
    10: "Basement",
}

FILL = ["回填", "填土", "表土", "雜填", "瀝青", "混凝土", "水泥", "級配", "路基", "碎石級配", "植生", "AC"]
HARD_ROCK = ["板岩", "片岩", "變質", "大理岩", "石英岩", "片麻岩", "安山岩", "玄武岩",
             "花崗岩", "輝綠岩", "火成岩", "千枚岩", "硬頁岩"]
SOFT_ROCK = ["石灰岩", "泥岩", "頁岩", "砂岩", "粉砂岩", "礫岩", "凝灰岩", "泥質砂岩", "砂質泥岩",
             "岩盤", "風化岩", "軟岩", "膠結"]
INTERBED = ["互層", "夾層", "薄層", "交替"]
CLAYSILT = ["黏土", "粘土", "粉土", "泥土", "壤土", "泥", "黏", "粉"]
SAND = ["砂"]
GRAVEL = ["礫", "卵石", "圓礫", "塊石", "卵礫"]

# L2 專用關鍵字
BASEMENT_ROCK = ["板岩", "片岩", "變質", "大理岩", "石英岩", "片麻岩", "安山岩", "玄武岩", "花崗岩", "輝綠岩", "火成岩", "基盤"]
SANDSTONE_ROCK = ["砂岩", "粉砂岩", "礫岩", "石灰岩", "凝灰岩"]
MUDSTONE_ROCK = ["泥岩", "頁岩"]
COARSE_SAND = ["粗砂", "中砂", "極粗砂"]
FINE_SAND = ["細砂", "極細砂"]
SILT = ["粉土", "粉砂"]
CLAY = ["黏土", "粘土", "泥土", "泥", "黏", "粉質黏土"]

# 次要成分：含 / 夾 / 少量 + 至多 4 字的材料名
_MINOR = re.compile(r"(?:含|夾|偶夾|微含|略含|少量|混|帶)[^，,。;；、\s（）()]{0,4}")
# X質 修飾詞：粉質、砂質、黏質、礫質、泥質
_MODIFIER = re.compile(r"[粉砂黏粘礫泥壤]質")
_SOILS_L1 = [("Gravel", GRAVEL), ("Sand", SAND), ("ClaySilt", CLAYSILT)]


def _main_soil_l1(text: str) -> int:
    t = _MODIFIER.sub("", _MINOR.sub("", text))
    best, pos = 0, 10**9
    for name, kws in _SOILS_L1:
        for kw in kws:
            i = t.find(kw)
            if 0 <= i < pos:
                pos = i
                best = {"Gravel": 4, "Sand": 3, "ClaySilt": 2}[name]
    return best


def classify_l1(desc: str | None) -> int:
    """經典 L1 7 類分類法 (向後相容)。"""
    if not desc:
        return 0
    t = str(desc).strip()
    if not t:
        return 0
    if any(k in t for k in FILL) and not any(k in t for k in ("岩",)):
        return 1
    if any(k in t for k in HARD_ROCK):
        return 7
    if any(k in t for k in INTERBED):
        return 5
    if any(k in t for k in SOFT_ROCK) or re.search(r"[泥砂頁礫]岩", t):
        return 6
    return _main_soil_l1(t)


def classify_l2(desc: str | None) -> int:
    """L2 10 類地質細化分類法。"""
    if not desc:
        return 0
    t = str(desc).strip()
    if not t:
        return 0

    # 1. 人工回填土
    if any(k in t for k in FILL) and not any(k in t for k in ("岩",)):
        return 1

    # 2. 岩盤判別
    if any(k in t for k in BASEMENT_ROCK):
        return 10
    if any(k in t for k in SANDSTONE_ROCK) or re.search(r"[砂礫]岩", t):
        return 9
    if any(k in t for k in MUDSTONE_ROCK) or re.search(r"[泥頁]岩", t):
        return 8

    # 3. 互層與過渡相 (砂泥混合、互層) 判別
    if any(k in t for k in INTERBED):
        return 6
    if any(k in t for k in ["砂質泥", "泥質砂", "砂泥", "泥砂", "砂質粉土"]):
        return 6

    # 4. 移除修飾詞與次要成分後，取主體土壤名詞
    t_clean = _MODIFIER.sub("", _MINOR.sub("", t))

    # 礫石優先
    if any(k in t_clean for k in GRAVEL):
        return 7

    # 粉砂 / 粉土 (需優先於砂，避免「粉砂」中的「砂」先匹配)
    if any(k in t_clean for k in SILT):
        return 3

    # 細砂
    if any(k in t_clean for k in FINE_SAND):
        return 4

    # 中粗砂
    if any(k in t_clean for k in COARSE_SAND):
        return 5

    # 剩餘砂
    if "砂" in t_clean:
        return 5

    # 黏土 / 泥
    if any(k in t_clean for k in CLAY):
        return 2

    return 0


def classify(desc: str | None, level: str = "l2") -> int:
    """統一分類介面，預設 level='l2'。可指定 level='l1'。"""
    if level == "l1":
        return classify_l1(desc)
    return classify_l2(desc)
