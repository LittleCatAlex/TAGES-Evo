"""5090 外接硬碟精簡搬運包建立工具 (Package for RTX 5090 Standalone Run)。

功能：
1. 依據 data/interim/chiayi_manifest.csv，只複製嘉義有配對到的 2,099 張柱狀圖圖檔，
   避免複製全台 75,212 張 (55 GB) 浪費隨身硬碟空間與傳輸時間。
2. 複製專案程式碼、設定檔、資料庫中間檔與一鍵執行腳本。
3. 建立 5090 環境檢驗 (check_env.bat) 與超時防呆排程 (run_on_5090.bat)。
"""
import argparse
import os
import shutil
from pathlib import Path
import pandas as pd

CHECK_ENV_BAT = """@echo off
echo ===================================================
echo   TAGES RTX 5090 運行環境自動檢驗
echo ===================================================
echo [1] 檢查 Python 版本:
python --version
echo.
echo [2] 檢查 GPU 與 CUDA 支援:
python -c "import torch; print('CUDA 可用性:', torch.cuda.is_available()); print('GPU 型號:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未檢測到 GPU'); print('CUDA 版本:', torch.version.cuda)" 2>nul
if errorlevel 1 (
    echo [提示] 尚未安裝 PyTorch，請先執行: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
)
echo.
echo [3] 檢查 OCR 與核心套件:
python -c "import rapidocr_onnxruntime, cv2, pandas, sklearn; print('核心套件全部就緒！')" 2>nul
if errorlevel 1 (
    echo [提示] 尚未安裝 OCR 套件，請執行: pip install rapidocr-onnxruntime opencv-python-headless opencc-python-reimplemented
)
echo ===================================================
pause
"""

RUN_5090_BAT = """@echo off
echo ===================================================
echo   TAGES RTX 5090 批次任務排程器 (上限 7.5 小時防呆)
echo ===================================================
echo 開始時間: %date% %time%

REM 執行優先級 1 (深孔 >=100m) 與優先級 2 (>=60m) 之批次 OCR
python scripts\\run_ocr_pipeline.py --manifest data\\interim\\chiayi_manifest.csv --photos data\\chart_photos --out data\\interim\\chiayi_ocr_results.jsonl

echo.
echo ===================================================
echo 任務完成！結束時間: %date% %time%
echo 結果已寫入: data\\interim\\chiayi_ocr_results.jsonl
echo ===================================================
pause
"""

README_5090_MD = """# TAGES RTX 5090 離線作業包操作說明

## 🎯 任務目標
在 RTX 5090 高效能機器上，批次執行嘉義 2,099 口工程鑽孔柱狀圖之 OCR 與地層解析，產出 `chiayi_ocr_results.jsonl`。

## 🚀 執行步驟（免聯網 / 快速聯網均可）

### 第一步：環境檢驗
- 雙擊執行 `check_env.bat`。
- 若顯示缺少 OCR 套件，請在聯網環境執行：
  ```cmd
  pip install rapidocr-onnxruntime opencv-python-headless opencc-python-reimplemented
  ```
  （5090 可直接聯網，下載僅需數十秒）。

### 第二步：啟動批次解析
- 雙擊執行 `run_on_5090.bat`。
- 程式內建自動斷點續跑（每解析成功一口井即自動追加寫入 `chiayi_ocr_results.jsonl`）。
- 內建最長 7.5 小時超時中斷保護，符合單次上機 8 小時限制。

### 第三步：結果回收
- 任務完成後，只需將 `data\\interim\\chiayi_ocr_results.jsonl`（大小約數 MB）複製回隨身碟，帶回本地開發主機即可！
"""

def make_package(target_dir: str = "D:/tages_5090_pack", source_photos: str = "H:/地科科展/chart_photos", manifest_csv: str = "data/interim/chiayi_manifest.csv", max_photos: int | None = None):
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"正在打包 5090 外接硬碟專用資料夾: {dest}")
    
    # 1. 寫入批次檔與說明文件
    (dest / "check_env.bat").write_text(CHECK_ENV_BAT, encoding="ansi")
    (dest / "run_on_5090.bat").write_text(RUN_5090_BAT, encoding="ansi")
    (dest / "README_5090.md").write_text(README_5090_MD, encoding="utf-8")
    print("已建立 check_env.bat, run_on_5090.bat 與 README_5090.md")
    
    # 2. 複製專案原始碼 (src, scripts, configs, tests)
    for folder in ["src", "scripts", "tests", "docs", "models"]:
        src_f = Path(folder)
        dst_f = dest / folder
        if src_f.exists():
            shutil.copytree(src_f, dst_f, dirs_exist_ok=True)
            print(f"已複製資料夾: {folder}")
            
    # 3. 複製重要配置檔
    for f in ["pyproject.toml", ".gitignore"]:
        if Path(f).exists():
            shutil.copy2(f, dest / f)
            
    # 4. 複製中間資料表
    dest_data_interim = dest / "data" / "interim"
    dest_data_interim.mkdir(parents=True, exist_ok=True)
    for f in ["chiayi_manifest.csv", "chiayi_velocity_3d.csv", "hydro_boreholes.csv", "hydro_strata.csv"]:
        src_f = Path("data/interim") / f
        if src_f.exists():
            shutil.copy2(src_f, dest_data_interim / f)
            print(f"已複製資料表: {f}")
            
    # 5. 複製嘉義柱狀圖圖檔 (僅限 manifest 中有配對者)
    dest_photos = dest / "data" / "chart_photos"
    dest_photos.mkdir(parents=True, exist_ok=True)
    
    manifest_path = Path(manifest_csv)
    if manifest_path.exists() and Path(source_photos).exists():
        df = pd.read_csv(manifest_path, low_memory=False)
        charts = df[df.has_chart == True]["chart_file"].dropna().unique()
        if max_photos:
            charts = charts[:max_photos]
        print(f"開始篩選複製嘉義柱狀圖圖檔 (共 {len(charts)} 張)...")
        copied, missing = 0, 0
        src_p = Path(source_photos)
        for c in charts:
            src_file = src_p / c
            if src_file.exists():
                dst_file = dest_photos / c
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                copied += 1
            else:
                missing += 1
        print(f"圖檔複製完成！成功: {copied} 張, 找不到: {missing} 張 (大小約: {sum(f.stat().st_size for f in dest_photos.glob('*.png')) / (1024*1024):.1f} MB)")
        
    print(f"\n===================================================")
    print(f"5090 外接硬碟包製作完成！請將 {dest} 複製到隨身碟。")
    print(f"===================================================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/tages_5090_pack", help="外接硬碟目標路徑")
    ap.add_argument("--photos", default="H:/地科科展/chart_photos", help="H 槽照片目錄")
    ap.add_argument("--max-photos", type=int, default=None, help="最多複製圖檔數 (測試用)")
    args = ap.parse_args()
    make_package(target_dir=args.out, source_photos=args.photos, max_photos=args.max_photos)
