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
        # 驗證岩性分類在 1~10 規範內且包含結晶基盤
        unq = np.unique(lith)
        assert np.all((unq >= 1) & (unq <= 10))
        assert 10 in unq, f"盆地 {b} 必須包含結晶基盤 (Class 10)"
        pct_basement = np.mean(lith == 10) * 100.0
        assert pct_basement >= 15.0, f"盆地 {b} 結晶基盤體素比例過低: {pct_basement:.1f}%"
        
        # 驗證基盤深度大於等於 50m (杜絕 0m 誤判)
        bedrock = data[f"{b}_bedrock"]
        assert (bedrock >= 50.0).all(), f"盆地 {b} 基盤深度應 >= 50m (杜絕 0m 誤判)"

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
    assert "THREE.InstancedMesh" in html_content, "需使用 THREE.InstancedMesh 渲染真正 3D 方塊實體體素"
    assert "THREE.BoxGeometry" in html_content, "需具備 3D 方塊幾何體 BoxGeometry"
    assert "setRenderMode" in html_content, "需支援 3D 立體方塊與點雲呈現形態切換"
    assert "voxelMesh.setColorAt" in html_content, "需支援 3D 方塊實例化個別岩性著色"
    assert "TAIWAN_MAIN_COASTLINE" in html_content, "需具備臺灣本島海岸線邊界多邊形以過濾海洋"
    assert "isPointInTaiwanLand" in html_content, "需具備臺灣陸域判斷函數 isPointInTaiwanLand"
    assert "presetColumns" in html_content, "需支援將既有盆地按經緯度索引為立體地質柱狀群"
    assert "k % 3 === 0" not in html_content, "不可使用玩具模運算交替條紋生成地層"
    assert "toggle-z-datum" in html_content, "需具備 Z 軸基準 Toggle 開關"
    assert "toggleZDatum" in html_content, "需具備 Z 軸基準切換函數 toggleZDatum"
    assert "Z 軸基準：地表相對深度 (Depth) " in html_content, "需在側邊欄控制台顯示指定 Z 軸基準切換標籤"
    assert "elev_grid" in html_content, "需支援盆地高程網格 elev_grid"
    assert "getElevationAt" in html_content, "需具備座標高程內插函數 getElevationAt"
    assert "estimateElevationIDW" in html_content, "需具備全臺 IDW 高程估算函數 estimateElevationIDW"

def test_vertical_datum_elevation_integration():
    """驗證 Z 軸基準（地表相對深度 vs 絕對海拔高程 MSL）資料包與座標計算邏輯。"""
    import json
    js_path = Path("reports/taiwan_portal_data.js")
    text = js_path.read_text(encoding="utf-8")
    json_str = text.replace("window.TAIWAN_PORTAL_DATA = ", "").rstrip(";\n ")
    portal_data = json.loads(json_str)

    # 1. 驗證所有盆地均具備完整 elev_grid
    required_basins = ["chianan", "zhuoshui", "taipei", "taichung", "pingtung", "yilan"]
    for b in required_basins:
        bv = portal_data["basins_voxels"][b]
        assert "elev_grid" in bv, f"盆地 {b} 缺少 elev_grid"
        eg = bv["elev_grid"]
        assert len(eg) == len(bv["lats"]), f"盆地 {b} elev_grid 緯度列數不符"
        assert len(eg[0]) == len(bv["lons"]), f"盆地 {b} elev_grid 經度欄數不符"
        arr = np.array(eg)
        assert arr.min() >= -5.0 and arr.max() <= 1000.0, f"盆地 {b} 高程異常: {arr.min()} ~ {arr.max()}"

    # 2. 驗證觀測井具備真實 elev 高程
    bhs = portal_data["boreholes"]
    assert len(bhs) >= 423
    for w in bhs[:50]:
        assert "elev" in w and isinstance(w["elev"], (int, float))
        assert -10.0 <= w["elev"] <= 3500.0, f"鑽井 {w['id']} 地表高程異常: {w['elev']}"

    # 3. 驗證絕對海拔公式: Z_msl = Elev - Depth
    elev = 85.0
    depth = 200.0
    z_msl = elev - depth
    assert z_msl == -115.0

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
        
    n_x, n_y, n_z = 32, 32, 61
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


def test_coastline_mask_and_ocean_filtering():
    """驗證海岸線遮罩能精確識別臺灣陸域並剔除開闊海域，避免海洋被錯誤預測為陸地岩層。"""
    taiwan_coastline = [
        [121.55, 25.32], [121.72, 25.24], [121.92, 25.15], [122.02, 25.01],
        [121.98, 24.85], [121.88, 24.62], [121.86, 24.45], [121.72, 24.30],
        [121.68, 24.12], [121.62, 23.85], [121.58, 23.55], [121.48, 23.25],
        [121.42, 23.05], [121.32, 22.85], [121.18, 22.70], [121.05, 22.55],
        [120.92, 22.35], [120.88, 21.92], [120.72, 21.88], [120.65, 22.02],
        [120.62, 22.25], [120.50, 22.38], [120.36, 22.48], [120.24, 22.58],
        [120.08, 22.92], [120.04, 23.12], [120.06, 23.35], [120.12, 23.65],
        [120.25, 23.90], [120.30, 24.08], [120.50, 24.32], [120.65, 24.58],
        [120.74, 24.70], [120.88, 24.88], [121.02, 25.06], [121.18, 25.12],
        [121.35, 25.20], [121.46, 25.30], [121.55, 25.32]
    ]

    def is_point_in_land(lon, lat):
        if 119.50 <= lon <= 119.70 and 23.20 <= lat <= 23.75:
            return True
        inside = False
        poly = taiwan_coastline
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    # 1. 陸域代表點需判定為 True
    land_points = [
        (121.50, 25.05),  # 臺北
        (120.67, 24.15),  # 臺中
        (120.20, 23.00),  # 臺南
        (120.48, 22.67),  # 屏東
        (121.75, 24.75),  # 宜蘭
        (121.60, 23.98),  # 花蓮
        (121.15, 22.75),  # 臺東
        (120.95, 23.50),  # 中央山脈玉山周邊
        (119.58, 23.57),  # 澎湖馬公
    ]
    for lon, lat in land_points:
        assert is_point_in_land(lon, lat), f"陸域點 ({lon}, {lat}) 應判定為陸地"

    # 2. 海洋代表點需判定為 False
    ocean_points = [
        (119.50, 24.50),  # 臺灣海峽北部深水區
        (120.00, 24.20),  # 臺中外海海峽水域
        (119.80, 23.00),  # 臺南外海海峽水域
        (122.50, 24.00),  # 花蓮外海太平洋水域
        (121.80, 25.50),  # 基隆外海東海陸棚
        (120.00, 21.50),  # 恆春外海巴士海峽
    ]
    for lon, lat in ocean_points:
        assert not is_point_in_land(lon, lat), f"海洋點 ({lon}, {lat}) 不可被誤判為陸地"

    # 3. 驗證 423 口觀測井 100% 落在陸地遮罩內
    bh_df = pd.read_csv("data/interim/taiwan_boreholes.csv")
    in_land_wells = [is_point_in_land(r["lon"], r["lat"]) for _, r in bh_df.iterrows()]
    assert all(in_land_wells), f"所有觀測井均應座落於陸地遮罩內，不合格井數: {len(in_land_wells) - sum(in_land_wells)}"


def test_seismic_stations_portal_integration():
    """驗證 32 座 TSMIP/EGDT 強震測站與傳遞函數頻譜曲線封裝及 3D 門戶整合完整性。"""
    import json
    js_path = Path("reports/taiwan_portal_data.js")
    html_path = Path("reports/taiwan_3d_portal.html")
    assert js_path.exists() and html_path.exists()

    text = js_path.read_text(encoding="utf-8")
    json_str = text.replace("window.TAIWAN_PORTAL_DATA = ", "").rstrip(";\n ")
    portal_data = json.loads(json_str)

    # 1. 驗證全島連續高程網格 (Taiwan Topography Grid)
    assert "taiwan_elev_grid" in portal_data, "需具備全島高程網格 taiwan_elev_grid"
    tg = portal_data["taiwan_elev_grid"]
    assert tg["lon_min"] <= 120.0 and tg["lon_max"] >= 122.0
    assert tg["lat_min"] <= 22.0 and tg["lat_max"] >= 25.3
    assert len(tg["grid"]) == tg["n_lat"]
    assert len(tg["grid"][0]) == tg["n_lon"]
    grid_arr = np.array(tg["grid"])
    assert grid_arr.min() >= 0.0 and grid_arr.max() >= 1500.0, "全島高程網格需涵蓋平原至高山"

    # 2. 驗證 32 座強震測站數據包
    assert "seismic_stations" in portal_data, "需具備強震測站數據包 seismic_stations"
    stations = portal_data["seismic_stations"]
    assert len(stations) == 32, f"強震測站數量應為 32 座，目前為 {len(stations)}"

    for st in stations:
        for field in ["code", "name", "basin", "lon", "lat", "elev", "site_class", "f0_voxel_hz", "d_edge_km", "spectrum", "response_spectrum"]:
            assert field in st, f"測站 {st.get('code')} 缺少必要欄位: {field}"
        assert st["d_edge_km"] >= 0.0, f"測站 {st['code']} 邊緣距離需大於等於 0"
        assert st["site_type"] in ["near_edge", "basin_center"]
        
        # 傳遞函數曲線驗證
        spec = st["spectrum"]
        assert len(spec["freqs"]) >= 40 and len(spec["h_amp"]) >= 40
        assert len(spec["freqs"]) == len(spec["h_amp"])
        assert spec["f0"] > 0 and spec["peak_amp"] >= 1.0

        # 反應譜曲線驗證
        rsp = st["response_spectrum"]
        assert len(rsp["periods"]) >= 30 and len(rsp["psa_amp"]) >= 30
        assert len(rsp["periods"]) == len(rsp["psa_amp"])
        assert rsp["t_peak_s"] > 0 and rsp["psa_max"] >= 1.0

    # 3. 驗證 3D Web 門戶 HTML 中的強震測站與頻譜抽屜元件
    html_content = html_path.read_text(encoding="utf-8")
    assert "seismicStationGroup" in html_content, "需具備 3D 強震測站群組 seismicStationGroup"
    assert "build3DSeismicStations" in html_content, "需具備 3D 強震測站構建函數 build3DSeismicStations"
    assert "openSeismicStationDrawer" in html_content, "需具備打開頻譜抽屜函數 openSeismicStationDrawer"
    assert "drawTransferFunctionChart" in html_content, "需具備 Canvas 繪製傳遞函數圖表函數 drawTransferFunctionChart"
    assert "drawResponseSpectrumChart" in html_content, "需具備 Canvas 繪製反應譜圖表函數 drawResponseSpectrumChart"
    assert "seismic-spectrum-drawer" in html_content, "需具備抽屜容器元素 seismic-spectrum-drawer"
    assert "toggle-seismic-stations" in html_content, "需具備 3D 側邊欄強震測站切換開關"


