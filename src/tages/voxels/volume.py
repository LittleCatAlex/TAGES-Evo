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
    vs_mudstone_thresh: float = 2150.0   # 泥岩/頁岩軟岩基底 (m/s)
    vs_sandstone_thresh: float = 2500.0  # 固結砂岩/礫岩 (m/s)
    vs_basement_thresh: float = 2800.0   # 堅硬結晶基底 (m/s)
    vs_acoustic_bedrock: float = 2150.0  # 地震聲學基盤 (Vs >= 2,150 m/s，進入第三紀半固結/固結泥岩砂岩基底)


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
            from tages.models.trainer import WellboreSeqTrainer
            self.seq_model, self.extractor, self.ckpt = WellboreSeqTrainer.load_checkpoint(self.model_path)
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

        # 6. 地球物理與深部地質約束層 (Geophysical Stratigraphic Constraint)
        # 標記推論範圍：0~300m 為鑽孔檢驗區 (Z1/Z2)，300~1000m 為外推與基盤區 (Z3)
        is_extrapolated = (flat_depth > 300.0)

        # 深部岩性判別：當深度 > 250m 且波速達到軟岩/硬岩基底標準時，啟動細化岩盤分類
        # Class 8 (泥岩/頁岩: Mudstone): 2150 <= Vs < 2500
        # Class 9 (砂岩/礫岩: Sandstone): 2500 <= Vs < 2800
        # Class 10 (堅硬結晶基底: Basement): Vs >= 2800
        mask_mudstone = (flat_depth > 250.0) & (vs >= cfg.vs_mudstone_thresh) & (vs < cfg.vs_sandstone_thresh)
        mask_sandstone = (flat_depth > 350.0) & (vs >= cfg.vs_sandstone_thresh) & (vs < cfg.vs_basement_thresh)
        mask_basement = (flat_depth > 450.0) & (vs >= cfg.vs_basement_thresh)
        
        pred_lith[mask_mudstone] = 8
        pred_lith[mask_sandstone] = 9
        pred_lith[mask_basement] = 10

        # 塑造成 3D 體素形狀
        lith_3d = pred_lith.reshape(shape_3d)
        conf_3d = pred_conf.reshape(shape_3d)
        entropy_3d = entropy.reshape(shape_3d)
        vs_3d = vs.reshape(shape_3d)
        vp_3d = vp.reshape(shape_3d)
        z_msl_3d = flat_z_msl.reshape(shape_3d)
        extra_3d = is_extrapolated.reshape(shape_3d)

        # 7. 萃取 2D 聲學基底深度 (Acoustic Bedrock Surface Z_bedrock)
        # 沿垂直軸尋找第一個 Vs >= vs_acoustic_bedrock 的深度
        bedrock_depth_2d = np.full((cfg.n_lat, cfg.n_lon), fill_value=cfg.depth_max, dtype=float)
        for i in range(cfg.n_lat):
            for j in range(cfg.n_lon):
                vs_col = vs_3d[i, j, :]
                match_indices = np.where(vs_col >= cfg.vs_acoustic_bedrock)[0]
                if len(match_indices) > 0:
                    bedrock_depth_2d[i, j] = depths[match_indices[0]]

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
            "config": cfg
        }

    def save_volume(self, data: dict[str, np.ndarray], out_path: str | Path):
        """儲存 3D 體素資料至壓縮 NPZ。"""
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            out_file,
            lats=data["lats"],
            lons=data["lons"],
            depths=data["depths"],
            lithology_3d=data["lithology_3d"].astype(np.int8),
            confidence_3d=data["confidence_3d"].astype(np.float32),
            entropy_3d=data["entropy_3d"].astype(np.float32),
            vs_3d=data["vs_3d"].astype(np.float32),
            vp_3d=data["vp_3d"].astype(np.float32),
            z_msl_3d=data["z_msl_3d"].astype(np.float32),
            is_extrapolated_3d=data["is_extrapolated_3d"].astype(bool),
            bedrock_depth_2d=data["bedrock_depth_2d"].astype(np.float32)
        )
        print(f"3D 體素地體模型已儲存至: {out_file} (檔案大小約 {out_file.stat().st_size / 1024 / 1024:.2f} MB)")
