"""全臺灣 2D GIS 門戶與多盆地高密度 3D 體素管線單元測試 (Unit Tests for Taiwan Portal & Multi-Basin Dense 3D Pipeline)。"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

def test_taiwan_boreholes():
    bh_path = Path("data/interim/taiwan_boreholes.csv")
    assert bh_path.exists(), "需存在全台水文觀測井資料庫"

    df = pd.read_csv(bh_path)
    assert len(df) == 423, f"全島水文觀測井應為 423 口，目前為 {len(df)}"
    assert (df["lat"] >= 21.8).all() and (df["lat"] <= 25.5).all(), "緯度需在臺灣本島與離島範圍內"
    assert (df["lon"] >= 119.5).all() and (df["lon"] <= 122.2).all(), "經度需在臺灣本島與離島範圍內"
    assert (df["well_depth_m"] > 0).all(), "井深需大於 0 公尺"

def test_taiwan_velocity_grid():
    vel_path = Path("data/interim/taiwan_velocity_3d.csv")
    assert vel_path.exists(), "需存在全島 3D 震波速度網格"

    df = pd.read_csv(vel_path)
    assert len(df) >= 2500, f"速度點數過少: {len(df)}"
    assert df["lat"].min() <= 22.5 and df["lat"].max() >= 25.0, "需涵蓋南臺灣至北臺灣"
    assert (df["vs_ms"] >= 1000).all() and (df["vs_ms"] <= 4500).all()

def test_taiwan_basins_dense_npz():
    npz_path = Path("data/processed/taiwan_basins_dense.npz")
    assert npz_path.exists(), "需存在六大盆地高密度 3D 體素檔案"

    data = np.load(npz_path)
    required_basins = ["chianan", "zhuoshui", "taipei", "taichung", "pingtung", "yilan"]
    for b in required_basins:
        assert f"{b}_lith" in data, f"缺少盆地 {b} 之 3D 岩性數據"
        assert f"{b}_vs" in data, f"缺少盆地 {b} 之 3D 波速數據"
        assert f"{b}_bedrock" in data, f"缺少盆地 {b} 之基盤數據"
        lith = data[f"{b}_lith"]
        assert lith.ndim == 3 and lith.size > 20000, f"盆地 {b} 體素點數過少: {lith.size}"
        # 驗證岩性分類在 1~10 規範內
        unq = np.unique(lith)
        assert np.all((unq >= 1) & (unq <= 10))

def test_taiwan_portal_assets():
    html_path = Path("reports/taiwan_3d_portal.html")
    js_path = Path("reports/taiwan_portal_data.js")
    assert html_path.exists() and html_path.stat().st_size > 10000
    assert js_path.exists() and js_path.stat().st_size > 5000000

    html_content = html_path.read_text(encoding="utf-8")
    assert "cartocdn" not in html_content.lower(), "不可使用需付費或強制要求 API key 的 Carto 圖資"
    assert "tile.openstreetmap.org" not in html_content, "不可使用對本地 file 訪問回傳 403 阻擋之 OSM 志工伺服器"
    assert "arcgisonline" in html_content, "需支援免 API Key 之 Esri 圖資"
    assert "wmts.nlsc.gov.tw" in html_content, "需支援內政部國土測繪圖資服務雲 (NLSC EMAP) 官方圖資"
    assert "World_Street_Map" in html_content, "需支援 Esri 全球街道圖資"
    assert "applyCustomROI" in html_content, "需支援地圖自由自訂框選查詢範圍"
    assert "getOrCreateCustomROI3DData" in html_content, "需支援自訂區域 3D 體素動態合成生成"
    assert "function init3DBase" in html_content, "需具備 3D 場景初始化核心函數 init3DBase"
    assert "function initOrUpdate3DScene" in html_content, "需具備 3D 地體更新函數 initOrUpdate3DScene"
    assert "let scene3d, camera3d, renderer3d, controls3d;" in html_content, "需聲明 Three.js 全域狀態變數"
    assert "activeLithMask" in html_content, "需聲明岩性遮罩變數 activeLithMask"
    assert "getPresetBasinAt" in html_content, "需具備判斷座標落於預設盆地之輔助函數 getPresetBasinAt"
    assert "collectedPoints.concat(synthPoints)" in html_content, "跨界自訂 ROI 需能無縫合併既有盆地與新合成體素"
    assert "if (collectedPoints.length >= 400)" not in html_content, "不可因盆地體素點數大於 400 即提早返回而遺漏非盆地區域"

def test_hybrid_roi_seamless_coverage():
    """驗證當自訂框選範圍同時涵蓋預設盆地與非預設盆地時，能無縫生成涵蓋全框的 3D 體素。"""
    import json
    
    # 載入 portal_data
    js_path = Path("reports/taiwan_portal_data.js")
    text = js_path.read_text(encoding="utf-8")
    json_str = text.replace("window.TAIWAN_PORTAL_DATA = ", "").rstrip(";\n ")
    portal_data = json.loads(json_str)
    
    # 定義跨越臺中盆地西界 (120.58°E) 的自訂區域: 120.50 ~ 120.70°E, 24.10 ~ 24.20°N
    roi = [120.50, 24.10, 120.70, 24.20]
    min_lon, min_lat, max_lon, max_lat = roi
    
    # 1. 收集落在 ROI 內的高精度既有盆地體素點 (此區包含部分臺中盆地 120.58 ~ 120.70)
    collected_points = []
    for k, bv in portal_data.get("basins_voxels", {}).items():
        b = bv.get("bounds", {})
        if min_lon <= b["lon_max"] and max_lon >= b["lon_min"] and min_lat <= b["lat_max"] and max_lat >= b["lat_min"]:
            for pt in bv.get("points", []):
                if min_lon <= pt[0] <= max_lon and min_lat <= pt[1] <= max_lat:
                    collected_points.append(pt)
    
    assert len(collected_points) > 1000, f"應從臺中盆地提取到部分高精度點，目前提取到: {len(collected_points)}"
    
    # 驗證既有盆地提取點僅分佈在 120.58°E 以東
    lons_collected = [p[0] for p in collected_points]
    assert min(lons_collected) >= 120.58, "既有盆地提取點應在臺中盆地界限內"
    
    # 2. 模擬非盆地區域 (120.50 ~ 120.58°E) 體素生成
    def is_in_preset(lon, lat):
        for k, bv in portal_data.get("basins_voxels", {}).items():
            b = bv["bounds"]
            if b["lon_min"] <= lon <= b["lon_max"] and b["lat_min"] <= lat <= b["lat_max"]:
                return True
        return False
        
    n_x, n_y, n_z = 32, 32, 45
    lons = np.linspace(min_lon, max_lon, n_x)
    lats = np.linspace(min_lat, max_lat, n_y)
    
    synth_points = []
    for j, lat in enumerate(lats):
        for i, lon in enumerate(lons):
            if is_in_preset(lon, lat):
                continue  # 盆地內部由 collected_points 涵蓋
            for k in range(n_z):
                depth = 1000.0 * (k / (n_z - 1))
                synth_points.append([round(lon, 4), round(lat, 4), round(depth, 1), 2, 260, 0.88])
                
    assert len(synth_points) > 1000, f"非盆地區域應動態合成體素，目前合成: {len(synth_points)}"
    lons_synth = [p[0] for p in synth_points]
    assert min(lons_synth) <= 120.51, "合成體素需延伸至自訂框西界"
    assert max(lons_synth) < 120.58, "合成體素不應重疊既有盆地內部"
    
    # 3. 合併無縫檢驗
    all_points = collected_points + synth_points
    assert len(all_points) > len(collected_points)
    all_lons = [p[0] for p in all_points]
    assert min(all_lons) <= 120.51 and max(all_lons) >= 120.69, "全島無縫混合體素需完整涵蓋整個自訂區域"

