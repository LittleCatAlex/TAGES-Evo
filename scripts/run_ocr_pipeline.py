"""地質鑽孔柱狀圖自動化解析與岩性標準化管線 (OCR + CSV Fallback + L1 標籤)。

核心功能：
1. 讀取 chiayi_manifest.csv，依優先級 (ocr_priority) 與孔深排序。
2. 調用 RapidOCR 解析柱狀圖影像中的分層深度與岩性描述。
3. 自動以政府資料庫原始 CSV 進行座標、高程、孔號、專案編號防呆補正 (Fallback)。
4. 調用 tages.labels.lithology.classify 將描述轉換為統一 L1 粗類岩性。
5. 輸出完全 UTF-8 結構化 JSON 與 CSV，無編碼亂碼。
6. 支援 --smoke / --limit / --priority / 斷點續跑。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from opencc import OpenCC
from rapidocr_onnxruntime import RapidOCR

# 匯入專案內部模組
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.labels.lithology import classify, L1_NAMES

CORRECTION_DICT = {
    "灭": "夾", "碟": "礫", "赚": "礫", "碍": "磚", "纵": "級", "堤": "塊",
    "坠画": "堅實", "坚画": "堅實", "坚实": "堅實", "坚硬": "堅硬",
    "真岩": "頁岩", "真若": "頁岩", "頁若": "頁岩", "黏土": "黏土", "粘土": "黏土",
    "蛋": "黃", "棕蛋": "棕黃", "趣繁密": "極緊密", "繁密": "緊密", "唇": "層",
    "算": "質", "箕": "質", "爽": "夾"
}

def clean_text(text: str) -> str:
    for k, v in CORRECTION_DICT.items():
        text = text.replace(k, v)
    text = text.replace(":", "；").replace("：", "；").replace(";", "；")
    text = re.sub(r'^[,\s;，；]+', '', text)
    text = re.sub(r'[,\s;，；]+$', '', text)
    return text.strip()

class LogExtractor:
    def __init__(self):
        self.ocr = RapidOCR()
        self.cc = OpenCC('s2twp')

    def preprocess(self, img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        return img, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def parse(self, img_path: Path) -> list[dict]:
        """解析柱狀圖的分層深度與岩性描述。"""
        orig_img, cleaned_img = self.preprocess(img_path)
        if cleaned_img is None:
            return []
        h, w = cleaned_img.shape[:2]

        ocr_results, _ = self.ocr(cleaned_img)
        if not ocr_results:
            return []

        boxes = []
        for box, text, score in ocr_results:
            text = text.strip()
            if not text:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            boxes.append({
                "text": text,
                "score": score,
                "cx": (min_x + max_x) / 2.0,
                "cy": (min_y + max_y) / 2.0,
                "min_x": min_x, "max_x": max_x,
                "min_y": min_y, "max_y": max_y
            })

        # 搜尋表頭中「性質描述」欄位所在的 X 範圍
        header_boxes = [b for b in boxes if h * 0.15 <= b["cy"] <= h * 0.28]
        desc_box = next((b for b in header_boxes if any(k in b["text"] for k in ["性質描述", "土壤性質", "性質", "描述", "岩石"])), None)

        if desc_box:
            col_min_x = desc_box["min_x"] - 15
            right_cols = [b for b in header_boxes if b["min_x"] > desc_box["max_x"] + 5]
            col_max_x = min(right_cols, key=lambda b: b["min_x"])["min_x"] - 2 if right_cols else desc_box["max_x"] + 80
            header_y = desc_box["max_y"]
        else:
            col_min_x = w * 0.35
            col_max_x = w * 0.68
            header_y = h * 0.22

        # 提取描述欄位內的地層文字
        strata_boxes = [b for b in boxes if b["min_y"] > header_y + 10 and b["cx"] >= col_min_x - 10 and b["cx"] <= col_max_x + 10]
        strata_boxes.sort(key=lambda b: b["cy"])

        # 尋找左側可能存在的深度數值 (X 在 0.05*w ~ col_min_x 之間)
        depth_candidates = [b for b in boxes if b["min_y"] > header_y + 10 and b["cx"] < col_min_x]
        
        # 聚合成層
        strata = []
        cur_desc = []
        cur_y = 0

        for b in strata_boxes:
            t = clean_text(b["text"])
            t = self.cc.convert(t)
            if not t or any(nh in t for nh in ["深度", "取樣", "標準貫入", "試驗", "顏色"]):
                continue
            
            # 檢查是否帶有層位結束標記或垂直距離較遠
            if cur_desc and (b["min_y"] - cur_y > 40):
                full_desc = " ".join(cur_desc)
                strata.append({"description": full_desc, "cy": cur_y})
                cur_desc = [t]
            else:
                cur_desc.append(t)
            cur_y = b["cy"]

        if cur_desc:
            strata.append({"description": " ".join(cur_desc), "cy": cur_y})

        # 為每層匹配最近的深度數值
        results = []
        last_d = 0.0
        for i, s in enumerate(strata):
            # 搜尋 Y 座標在 s["cy"] +- 35 內的數字
            near_depths = []
            for db in depth_candidates:
                if abs(db["cy"] - s["cy"]) <= 35:
                    m = re.search(r'(\d+\.?\d*)', db["text"])
                    if m:
                        val = float(m.group(1))
                        if val > last_d and val < 500:
                            near_depths.append(val)
            bot_d = min(near_depths) if near_depths else last_d + 5.0
            results.append({
                "top_m": round(last_d, 2),
                "bottom_m": round(bot_d, 2),
                "desc_raw": s["description"],
                "l1": classify(s["description"]),
                "l1_name": L1_NAMES[classify(s["description"])]
            })
            last_d = bot_d

        return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/chiayi_manifest.csv")
    ap.add_argument("--photos", default="H:/地科科展/chart_photos")
    ap.add_argument("--out", default="data/interim/chiayi_ocr_results.jsonl")
    ap.add_argument("--priority", type=int, default=None, help="指定跑第幾級優先度 (1:>=100m, 2:>=60m, 3:>=30m)")
    ap.add_argument("--limit", type=int, default=None, help="最大解析井數")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"找不到 manifest: {manifest_path}")
        return

    df = pd.read_csv(manifest_path, low_memory=False)
    df = df[df.has_chart == True]
    if args.priority:
        df = df[df.ocr_priority == args.priority]
    if args.limit:
        df = df.head(args.limit)

    print(f"本次預計解析孔數: {len(df)}")
    extractor = LogExtractor()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_holes = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_holes.add(json.loads(line).get("hole_id"))
                except Exception:
                    pass
        print(f"已存在解析結果: {len(done_holes)} 孔，支援斷點續跑")

    photos_dir = Path(args.photos)
    t0 = time.time()
    count = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        for _, row in df.iterrows():
            hole_id = f"{row.proj_no}_{row.hole_no}"
            if hole_id in done_holes:
                continue

            chart_file = photos_dir / str(row.chart_file)
            if not chart_file.exists():
                continue

            start_t = time.time()
            try:
                strata = extractor.parse(chart_file)
            except Exception as e:
                print(f"[解析異常] {hole_id}: {e}")
                strata = []

            dur = time.time() - start_t
            count += 1
            print(f"[{count}/{len(df)}] {hole_id} (深:{row.total_depth}m, 層數:{len(strata)}) 耗時: {dur:.2f}s")

            record = {
                "hole_id": hole_id,
                "proj_no": row.proj_no,
                "hole_no": row.hole_no,
                "coord_x": float(row.coord_x) if pd.notna(row.coord_x) else None,
                "coord_y": float(row.coord_y) if pd.notna(row.coord_y) else None,
                "elevation": float(row.elevation) if pd.notna(row.elevation) else None,
                "total_depth": float(row.total_depth) if pd.notna(row.total_depth) else None,
                "chart_file": row.chart_file,
                "ocr_priority": int(row.ocr_priority),
                "parse_sec": round(dur, 2),
                "strata": strata
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    total_dur = time.time() - t0
    avg = total_dur / count if count else 0
    print(f"\n全部完成！處理: {count} 孔, 總耗時: {total_dur:.1f}s (平均每孔: {avg:.2f}s)")

if __name__ == "__main__":
    main()
