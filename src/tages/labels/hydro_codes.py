"""水文地質資訊系統 TYPE_OF_ROCK 代碼 → L1 粗類 / L2 細化 10 類。

代碼來源：hydro.geologycloud.tw 前端 (map.js) 的 rockTable / rockTrans。
L2 類別：
  0 Unknown | 1 Fill | 2 ClayMud | 3 Silt | 4 FineSand | 5 CoarseSand | 6 SandMudMixed | 7 Gravel | 8 Mudstone | 9 Sandstone | 10 Basement
"""
from __future__ import annotations

FILL = {"tsh"}
SOFT_MUDSTONE = {"ms", "sh"}
SOFT_SANDSTONE = {"ss", "sls", "cgl", "tf", "agl", "ls", "ds"}
HARD_BASEMENT = {"br", "slt", "sch", "gns", "mbl", "and", "bs", "qtz", "spt"}
UNKNOWN = {"L", "dwd", ""}
PEAT = {"pt"}


def hydro_to_l2(code: str | None) -> int:
    """轉換為 L2 十大岩性類別。"""
    if code is None:
        return 0
    c = str(code).strip()
    if c in UNKNOWN:
        return 0
    if c in FILL:
        return 1
    if c in HARD_BASEMENT:
        return 10
    if c in SOFT_SANDSTONE:
        return 9
    if c in SOFT_MUDSTONE:
        return 8
    if c in PEAT:
        return 2

    # 砂泥混合 / 互層 (同時包含泥 M/C/Z 與砂 S)
    if ("M" in c and "S" in c) or ("Z" in c and "S" in c) or ("C" in c and "S" in c):
        return 6

    # 礫石 / 卵石
    if any(k in c for k in ("G", "B")):
        return 7

    # 細砂 / 極細砂
    if c in ("vfS", "fS"):
        return 4

    # 中粗砂
    if c in ("mS", "cS", "vcS", "S"):
        return 5

    # 粉砂 / 粉土
    if c == "Z" or (c.endswith("Z") and "S" not in c):
        return 3

    # 泥 / 黏土
    if c in ("C", "M") or (c.endswith("C") or c.endswith("M")):
        return 2

    last = c[-1]
    if last == "S":
        return 5
    if last in ("C", "M"):
        return 2
    if last == "Z":
        return 3

    return 0


def hydro_to_l1(code: str | None) -> int:
    """轉換為 L1 基礎 7 類。"""
    l2 = hydro_to_l2(code)
    mapping = {
        0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3,
        6: 5, 7: 4, 8: 6, 9: 6, 10: 7
    }
    return mapping.get(l2, 0)
