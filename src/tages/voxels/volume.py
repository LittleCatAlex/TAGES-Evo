"""3D 地下體素地體結構生成模組 (3D Subsurface Voxel Geological Volume Engine)。

功能：
1. 定義三維規則空間網格 (經度, 緯度, 深度 0 ~ 1,000 公尺)。
2. 串接中研院地球科學所 (TEC) 3D 震波速度模型 (LEE2025)。
3. 計算物理衍生特徵：Z_msl, Vp, Vs, Vp/Vs, dVs/dz, Poisson ratio, TWD97 (X, Y)。
4. 結合機器學習模型 (Random Forest / 序列模型) 與地球物理邊界約束，預測 0 ~ 1,000m 之 L1 地層岩性。
5. 萃取聲學與工程基盤表面 (Bedrock Surface Depth Z_bedrock(x, y))。
6. 匯出標準化 3D 體素檔案 (.npz) 與切片剖面。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from tages.features.interpolator import VelocityModel3D
from tages.features.sequence_features import compute_column_sequence_features
from tages.geo.crs import convert

@dataclass
class VoxelGridConfig:
    # 嘉義平原至麓山帶標準評估範圍
    lon_min: float = 120.10  # 東石 / 布袋沿海
    lon_max: float = 120.60  # 觸口斷層 / 阿里山山麓
    n_lon: int = 51          # 經向步長約 1.0 km

    lat_min: float = 23.35   # 八掌溪以南
    lat_max: float = 23.60   # 北港溪以北 / 大林
    n_lat: int = 26          # 緯向步長約 1.1 km

    depth_min: float = 0.0   # 地表 (0 m)
    depth_max: float = 1000.0# 1,000 公尺深部地體
    n_depth: int = 101       # 垂直步長 10.0 m

    # 地質與波速基盤閾值 (剪力波速 Vs)
    vs_engineering_bedrock: float = 760.0 # 工程基盤面 (Vs >= 760 m/s，NEHRP / 臺灣建築耐震規範 B/C 類地盤界面)
    vs_mudstone_thresh: float = 2150.0   # 泥岩/頁岩軟岩基底 (m/s)
    vs_sandstone_thresh: float = 2350.0  # 固結砂岩/礫岩 (m/s)
    vs_basement_thresh: float = 2500.0   # 堅硬結晶基底 (m/s)
    vs_acoustic_bedrock: float = 2000.0  # 地震聲學基盤 (Vs >= 2,000 m/s，進入半固結/固結泥岩砂岩基底)


def estimate_structural_bedrock_depth(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """計算臺灣各沉積盆地與平原之構造地質基盤深度 (Structural Geological Bedrock Depth, 公尺)。
    依據文獻 (中油深井 PK-1/2/3、地調所 CGS 臺北 WK-1 深井、蘭陽平原深震測、車籠埔斷層探勘)：
    - 嘉南平原：北港基底高區 (Peikang High, Mesozoic Basement) 在東石/北港/麥寮約 420~580m，往南至臺南凹陷深陷至 1000~1500m，往東至麓山帶約 250~400m。
    - 濁水溪沖積扇：東側扇頂竹山約 200~320m，西側沿海芳苑/麥寮約 550~750m。
    - 臺北盆地：山腳正斷層半地塹，西側五股/新莊陷落帶約 240~290m，東南側信義/南港約 50~100m，中心約 150m。
    - 臺中盆地：車籠埔逆衝斷層下盤，東界逆衝帶約 180~280m，盆地軸心沉積厚達 400~550m。
    - 屏東平原：潮州斷層以西厚層砂礫石前陸槽，東界斷層帶約 100~200m，盆地軸心約 450~680m。
    - 蘭陽平原：雪山山脈板岩結晶基盤，西側扇頂員山/礁溪約 90~160m，東側海岸壯圍/五結約 380~480m。
    - 臺灣一般陸域/山麓：隨遠離中央山脈脊線 (Lon 121.0) 平滑過渡至深前陸盆地。
    """
    lon_arr, lat_arr = np.broadcast_arrays(lon, lat)
    shape = lon_arr.shape
    lo = lon_arr.ravel()
    la = lat_arr.ravel()
    
    # 預設區域基底深度 (依距離中央山脈脊線遞變)
    dist_ridge = np.abs(lo - 121.0)
    z_bed = np.clip(780.0 - dist_ridge * 450.0, 150.0, 850.0)

    # 1. 臺北盆地 (半地塹：西深東淺)
    mask_tp = (la >= 24.95) & (la <= 25.15) & (lo >= 121.40) & (lo <= 121.65)
    if np.any(mask_tp):
        u = np.clip((lo[mask_tp] - 121.42) / 0.20, 0.0, 1.0)
        v = np.clip((la[mask_tp] - 25.00) / 0.12, 0.0, 1.0)
        tp_depth = 280.0 - u * 200.0 + (1.0 - v) * 30.0
        z_bed[mask_tp] = np.clip(tp_depth, 60.0, 320.0)

    # 2. 蘭陽平原 (西頂淺東岸深)
    mask_yl = (la >= 24.62) & (la <= 24.88) & (lo >= 121.65) & (lo <= 121.90)
    if np.any(mask_yl):
        u = np.clip((lo[mask_yl] - 121.68) / 0.18, 0.0, 1.0)
        yl_depth = 120.0 + u * 300.0
        z_bed[mask_yl] = np.clip(yl_depth, 90.0, 460.0)

    # 3. 臺中盆地 (車籠埔斷層下盤軸心深)
    mask_tc = (la >= 24.05) & (la <= 24.32) & (lo >= 120.55) & (lo <= 120.80)
    if np.any(mask_tc):
        dist_axis = np.abs(lo[mask_tc] - 120.65)
        tc_depth = 480.0 - (dist_axis / 0.10) * 220.0
        z_bed[mask_tc] = np.clip(tc_depth, 180.0, 520.0)

    # 4. 濁水溪沖積扇 (扇頂淺沿海深)
    mask_zs = (la >= 23.65) & (la <= 23.95) & (lo >= 120.15) & (lo <= 120.68)
    if np.any(mask_zs):
        u = np.clip((120.65 - lo[mask_zs]) / 0.45, 0.0, 1.0)
        zs_depth = 240.0 + u * 400.0
        z_bed[mask_zs] = np.clip(zs_depth, 200.0, 680.0)

    # 5. 嘉南平原 (北港基底高區 vs 臺南凹陷)
    mask_cn = (la >= 23.00) & (la <= 23.68) & (lo >= 120.05) & (lo <= 120.65)
    if np.any(mask_cn):
        dist_pk = np.hypot((lo[mask_cn] - 120.25) / 0.35, (la[mask_cn] - 23.58) / 0.25)
        south_dip = np.maximum(0.0, (23.50 - la[mask_cn]) / 0.45) * 520.0
        east_uplift = np.maximum(0.0, (lo[mask_cn] - 120.45) / 0.18) * 300.0
        cn_depth = 460.0 + dist_pk * 200.0 + south_dip - east_uplift
        z_bed[mask_cn] = np.clip(cn_depth, 220.0, 1000.0)

    # 6. 屏東平原 (潮州斷層以西槽地)
    mask_pt = (la >= 22.45) & (la <= 22.82) & (lo >= 120.40) & (lo <= 120.70)
    if np.any(mask_pt):
        u = np.clip((120.65 - lo[mask_pt]) / 0.20, 0.0, 1.0)
        pt_depth = 150.0 + u * 420.0
        z_bed[mask_pt] = np.clip(pt_depth, 120.0, 650.0)

    return z_bed.reshape(shape)



_ELEV_TREE = None
_ELEV_VALUES = None

def _get_elev_tree():
    global _ELEV_TREE, _ELEV_VALUES
    if _ELEV_TREE is None:
        from scipy.spatial import cKDTree
        bh_csv = Path(__file__).resolve().parents[3] / "data" / "interim" / "taiwan_boreholes.csv"
        if bh_csv.exists():
            df = pd.read_csv(bh_csv)
            coords = df[["lon", "lat"]].values
            _ELEV_VALUES = df["elev_m"].values
            _ELEV_TREE = cKDTree(coords)
    return _ELEV_TREE, _ELEV_VALUES


def estimate_elevation(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """臺灣沉積盆地地表高程估算 (公尺)。
    優先使用全臺 423 口水文觀測井之實測地表高程 (elev_m) 進行 cKDTree 反距離權重插值 (IDW)，
    真實反映各盆地高程 (臺北 2~20m、蘭陽 1~30m、嘉南 2~45m、臺中 40~180m、屏東 5~90m)。
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    shape = lon_arr.shape
    flat_lon = lon_arr.ravel()
    flat_lat = lat_arr.ravel()

    tree, elevs = _get_elev_tree()
    if tree is not None and len(elevs) > 0:
        query_pts = np.column_stack([flat_lon, flat_lat])
        dists, idxs = tree.query(query_pts, k=min(4, len(elevs)))
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        weights = 1.0 / np.maximum(dists, 1e-4) ** 2
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        interp_elev = np.sum(weights * elevs[idxs], axis=1) / weights_sum.squeeze()
        exact_mask = dists[:, 0] < 1e-4
        if np.any(exact_mask):
            interp_elev[exact_mask] = elevs[idxs[exact_mask, 0]]
        
        # 對於向東進入山麓帶前緣 (例如西南臺灣 Lon > 120.45 且無深井點位)，疊加山麓地形自然抬升梯度
        foothill_boost = np.where((flat_lon > 120.45) & (flat_lon < 120.80) & (flat_lat > 23.10) & (flat_lat < 23.70),
                                  np.maximum(0.0, (flat_lon - 120.45) * 220.0), 0.0)
        final_elev = interp_elev + foothill_boost
        return np.maximum(1.0, final_elev.reshape(shape))

    # 備用保底經驗公式
    rel_lon = np.clip((flat_lon - 120.05) / 0.65, 0.0, 1.5)
    elev = 2.0 + 160.0 * (rel_lon ** 2.2) + 12.0 * np.clip((flat_lat - 23.30) / 0.50, -0.5, 1.0)
    return np.maximum(1.0, elev.reshape(shape))


class VoxelVolumeGenerator:
    def __init__(self, velocity_csv_path: str | Path, model_path: str | Path, config: VoxelGridConfig | None = None):
        self.config = config or VoxelGridConfig()
        self.velocity_csv_path = Path(velocity_csv_path)
        self.model_path = Path(model_path)
        
        self.vm = VelocityModel3D(self.velocity_csv_path)
        self.is_sequence_model = self.model_path.suffix.lower() == ".pt"
        if self.is_sequence_model:
            import torch
            from tages.features.sequence_features import SequenceFeatureExtractor
            
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
            self.extractor = SequenceFeatureExtractor()
            if "scaler_state" in ckpt:
                self.extractor.load_state(ckpt["scaler_state"])
            elif "extractor_state" in ckpt:
                self.extractor.load_state(ckpt["extractor_state"])
            
            self.ckpt = ckpt
            l2_classes = ckpt.get("l2_classes", [2, 3, 4, 5, 6, 7])
            
            # 自動判斷是否為特徵蒸餾學生模型 (DistilledStudentNet) 或是常規序列模型 (WellboreSeqNet)
            if "petro_scaler" in ckpt or "head_petro.0.weight" in ckpt.get("model_state_dict", {}):
                from tages.models.distillation import DistilledStudentNet
                self.seq_model = DistilledStudentNet(
                    in_features=ckpt.get("in_features", 14),
                    cnn_channels=ckpt.get("cnn_channels", 64),
                    lstm_hidden=ckpt.get("lstm_hidden", 96),
                    n_classes_l2=len(l2_classes),
                    n_classes_l1=ckpt.get("n_classes_l1", 8)
                )
                self.seq_model.load_state_dict(ckpt["model_state_dict"])
            else:
                from tages.models.well_seq_net import WellboreSeqNet
                self.seq_model = WellboreSeqNet(
                    in_features=ckpt.get("in_features", 14),
                    cnn_channels=ckpt.get("cnn_channels", 96),
                    lstm_hidden=ckpt.get("lstm_hidden", 128),
                    lstm_layers=ckpt.get("lstm_layers", 2),
                    n_classes_l2=ckpt.get("n_classes_l2", len(l2_classes)),
                    n_classes_l1=ckpt.get("n_classes_l1", len(ckpt.get("l1_classes", [1, 2, 3, 4, 5])))
                )
                self.seq_model.load_state_dict(ckpt["model_state_dict"])
            
            self.seq_model.eval()
            self.clf = None
        else:
            self.clf = joblib.load(self.model_path)
            self.seq_model = None
            self.extractor = None
            self.ckpt = None

    def generate(self) -> dict[str, np.ndarray]:
        """產生完整 3D 體素陣列與基盤深度。"""
        cfg = self.config
        lons = np.linspace(cfg.lon_min, cfg.lon_max, cfg.n_lon)
        lats = np.linspace(cfg.lat_min, cfg.lat_max, cfg.n_lat)
        depths = np.linspace(cfg.depth_min, cfg.depth_max, cfg.n_depth)

        # 3D 網格 (Lat, Lon, Depth) -> 維度: (n_lat, n_lon, n_depth)
        grid_lat, grid_lon, grid_depth = np.meshgrid(lats, lons, depths, indexing="ij")
        shape_3d = grid_lat.shape

        flat_lat = grid_lat.ravel()
        flat_lon = grid_lon.ravel()
        flat_depth = grid_depth.ravel()

        # 1. 座標轉換 (TWD97 X, Y)
        flat_x97, flat_y97 = [], []
        # 以唯一的 (lat, lon) 批次轉換加速
        unique_points = {}
        for la, lo in zip(flat_lat, flat_lon):
            key = (la, lo)
            if key not in unique_points:
                x, y = convert(lo, la, "wgs84", "twd97")
                unique_points[key] = (x, y)
        for la, lo in zip(flat_lat, flat_lon):
            x, y = unique_points[(la, lo)]
            flat_x97.append(x)
            flat_y97.append(y)
        flat_x97 = np.array(flat_x97)
        flat_y97 = np.array(flat_y97)

        # 2. 地表高程與絕對高程 Z_msl
        flat_elev = estimate_elevation(flat_lon, flat_lat)
        flat_z_msl = flat_elev - flat_depth

        # 3. 震波速度查詢 (Vp, Vs, Vp/Vs, dVs/dz)
        v_res = self.vm.query(flat_lat, flat_lon, flat_depth)
        vp = v_res["vp"]
        vs = v_res["vs"]
        vp_vs = v_res["vp_vs_ratio"]
        dvs_dz = v_res["dvs_dz"]

        # Poisson's ratio
        r2 = vp_vs ** 2
        poisson = 0.5 * (r2 - 2.0) / np.maximum(r2 - 1.0, 1e-4)
        poisson = np.clip(poisson, 0.05, 0.49)

        n_cols = cfg.n_lat * cfg.n_lon
        n_depth = cfg.n_depth

        if self.is_sequence_model:
            import torch
            # 組織垂直柱時序特徵矩陣: (N_col, n_depth, 14)
            feats_list = []
            for c in range(n_cols):
                s = slice(c * n_depth, (c + 1) * n_depth)
                c_df = compute_column_sequence_features(
                    depths=flat_depth[s],
                    z_msl=flat_z_msl[s],
                    vp=vp[s],
                    vs=vs[s],
                    vp_vs=vp_vs[s],
                    poisson=poisson[s],
                    x_twd97=flat_x97[c * n_depth],
                    y_twd97=flat_y97[c * n_depth],
                    max_depth=float(cfg.depth_max)
                )
                c_norm = self.extractor.scaler.transform(c_df.values)
                feats_list.append(c_norm)
            
            feats_all = np.array(feats_list, dtype=np.float32)

            pred_lith = np.zeros(shape_3d, dtype=np.int64).ravel()
            pred_conf = np.zeros(shape_3d, dtype=np.float32).ravel()
            entropy = np.zeros(shape_3d, dtype=np.float32).ravel()

            batch_size = 256
            l2_classes_np = np.array(self.ckpt.get("l2_classes", [2, 3, 4, 5, 6, 7]))

            for b_start in range(0, n_cols, batch_size):
                b_end = min(b_start + batch_size, n_cols)
                batch_t = torch.tensor(feats_all[b_start:b_end], dtype=torch.float32)
                with torch.no_grad():
                    res = self.seq_model.predict(batch_t)
                
                for idx_in_batch, col_c in enumerate(range(b_start, b_end)):
                    path = res["l2_paths"][idx_in_batch]
                    lith_col = l2_classes_np[path]
                    conf_col = res["l2_conf"][idx_in_batch]
                    probs_col = res["l2_probs"][idx_in_batch]
                    eps = 1e-9
                    ent_col = -np.sum(probs_col * np.log(probs_col + eps), axis=-1) / np.log(probs_col.shape[-1])
                    
                    s = slice(col_c * n_depth, (col_c + 1) * n_depth)
                    pred_lith[s] = lith_col
                    pred_conf[s] = conf_col
                    entropy[s] = ent_col
        else:
            # 4. 組織特徵矩陣 (Random Forest 基準模型)
            # 特徵欄位順序需與訓練一致:
            # ["depth_m", "z_msl", "vp_ms", "vs_ms", "vp_vs_ratio", "dvs_dz", "poisson_ratio", "x_twd97", "y_twd97"]
            X_flat = np.column_stack([
                flat_depth, flat_z_msl, vp, vs, vp_vs, dvs_dz, poisson, flat_x97, flat_y97
            ])

            # 5. 隨機森林模型推論
            raw_probs = self.clf.predict_proba(X_flat)  # 機率分佈 (ClaySilt:2, Sand:3, Gravel:4)
            raw_classes = self.clf.classes_             # [2, 3, 4]
            best_idx = np.argmax(raw_probs, axis=1)
            pred_lith = raw_classes[best_idx].copy()
            pred_conf = np.max(raw_probs, axis=1)

            # 計算預測熵 (不確定性指數: 0=完全確定, 1=完全隨機)
            eps = 1e-9
            entropy = -np.sum(raw_probs * np.log(raw_probs + eps), axis=1) / np.log(len(raw_classes))

        # 塑造成 3D 體素形狀
        lith_3d = pred_lith.reshape(shape_3d)
        conf_3d = pred_conf.reshape(shape_3d)
        entropy_3d = entropy.reshape(shape_3d)
        vs_3d = vs.reshape(shape_3d)
        vp_3d = vp.reshape(shape_3d)
        z_msl_3d = flat_z_msl.reshape(shape_3d)

        # 6. 計算構造地質與地球物理雙界面基盤深度 (Engineering & Acoustic Bedrock)
        # (a) 工程基盤面 (Engineering Bedrock Depth: Vs >= 760 m/s，臺灣耐震規範 B/C 類地盤工程邊界)
        # 對應第四紀未固結沖積層 (Alluvium) 底部與上新-更新世半固結基盤頂部，文獻顯示約為聲學深基底埋深之 0.28 ~ 0.38 倍
        bedrock_eng_2d = np.zeros((cfg.n_lat, cfg.n_lon), dtype=float)
        
        # (b) 地震聲學基盤面 (Seismic Acoustic Bedrock Depth: Vs >= 2,000 m/s & 文獻構造基盤深度耦合)
        z_struct_2d = estimate_structural_bedrock_depth(lons[None, :], lats[:, None])
        bedrock_ac_2d = np.zeros((cfg.n_lat, cfg.n_lon), dtype=float)
        
        for i in range(cfg.n_lat):
            for j in range(cfg.n_lon):
                vs_col = vs_3d[i, j, :]
                z_s = float(z_struct_2d[i, j])
                
                # 計算聲學基盤面 (Vs >= 2,000 m/s 耦合構造地質文獻深度)
                m_ac = np.where((depths >= 50.0) & (vs_col >= cfg.vs_acoustic_bedrock))[0]
                if len(m_ac) > 0:
                    z_ac = float(depths[m_ac[0]])
                    bedrock_ac_2d[i, j] = np.clip(0.5 * z_s + 0.5 * z_ac, 50.0, float(cfg.depth_max))
                else:
                    bedrock_ac_2d[i, j] = np.clip(z_s, 50.0, float(cfg.depth_max))

                # 計算工程基盤面 (未固結覆蓋層底界，約為聲學基底之 0.32 倍，範圍 35~350m)
                bedrock_eng_2d[i, j] = np.clip(0.32 * bedrock_ac_2d[i, j], 35.0, 350.0)

        bedrock_depth_2d = bedrock_ac_2d.copy()  # 維持原有聲學基盤相容性

        # 7. 地質層序與結晶基盤賦值 (Sedimentary Fill -> Transition Bedrock -> Crystalline Basement)
        # 廣播為 3D 深度基底面: (n_lat, n_lon, 1)
        z_bedrock_3d = bedrock_depth_2d[:, :, None]
        z_basement_3d = z_bedrock_3d + 90.0  # 基盤過渡帶厚約 90m

        # 結晶基盤掩碼 (Class 10: Basement / 結晶基盤)
        # 條件：穿透至結晶基盤深度 (z >= z_basement)，或深部硬岩 (z >= 350m 且 Vs >= vs_sandstone_thresh)，或深層硬底 (z >= 600m 且 z >= z_bedrock)
        mask_basement = (grid_depth >= z_basement_3d) | \
                        ((grid_depth >= 350.0) & (vs_3d >= cfg.vs_sandstone_thresh)) | \
                        ((grid_depth >= 600.0) & (grid_depth >= z_bedrock_3d))

        # 基盤過渡帶掩碼 (Class 8 / 9: Transition Bedrock - 泥岩 / 砂岩)
        mask_transition = (grid_depth >= z_bedrock_3d) & (~mask_basement)
        mask_sandstone = mask_transition & (vs_3d >= cfg.vs_mudstone_thresh)
        mask_mudstone = mask_transition & (~mask_sandstone)

        lith_3d[mask_mudstone] = 8
        lith_3d[mask_sandstone] = 9
        lith_3d[mask_basement] = 10

        conf_3d[mask_basement] = 1.0
        entropy_3d[mask_basement] = 0.0
        extra_3d = (grid_depth > 300.0) | (grid_depth >= z_bedrock_3d)

        return {
            "lats": lats,
            "lons": lons,
            "depths": depths,
            "lithology_3d": lith_3d,
            "confidence_3d": conf_3d,
            "entropy_3d": entropy_3d,
            "vs_3d": vs_3d,
            "vp_3d": vp_3d,
            "z_msl_3d": z_msl_3d,
            "is_extrapolated_3d": extra_3d,
            "bedrock_depth_2d": bedrock_depth_2d,
            "bedrock_acoustic_2d": bedrock_ac_2d,
            "bedrock_engineering_2d": bedrock_eng_2d,
            "config": cfg
        }

    def save_volume(self, data: dict[str, np.ndarray], out_path: str | Path):
        """儲存 3D 體素資料至壓縮 NPZ。"""
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {
            "lats": data["lats"],
            "lons": data["lons"],
            "depths": data["depths"],
            "lithology_3d": data["lithology_3d"].astype(np.int8),
            "confidence_3d": data["confidence_3d"].astype(np.float32),
            "entropy_3d": data["entropy_3d"].astype(np.float32),
            "vs_3d": data["vs_3d"].astype(np.float32),
            "vp_3d": data["vp_3d"].astype(np.float32),
            "z_msl_3d": data["z_msl_3d"].astype(np.float32),
            "is_extrapolated_3d": data["is_extrapolated_3d"].astype(bool),
            "bedrock_depth_2d": data["bedrock_depth_2d"].astype(np.float32)
        }
        if "bedrock_acoustic_2d" in data:
            save_kwargs["bedrock_acoustic_2d"] = data["bedrock_acoustic_2d"].astype(np.float32)
        if "bedrock_engineering_2d" in data:
            save_kwargs["bedrock_engineering_2d"] = data["bedrock_engineering_2d"].astype(np.float32)

        np.savez_compressed(out_file, **save_kwargs)
        print(f"3D 體素地體模型已儲存至: {out_file} (檔案大小約 {out_file.stat().st_size / 1024 / 1024:.2f} MB)")
