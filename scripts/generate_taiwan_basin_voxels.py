"""臺灣主要盆地與平原高密度 3D 體素地體模型批次生成工具 (Taiwan Basin Dense Voxel Generator)。

生成全台六大主要地質平原/盆地之高密度 3D 體素模型：
1. 嘉南平原 (Chianan Plain)
2. 濁水溪沖積扇 (Zhuoshui River Fan)
3. 臺北盆地 (Taipei Basin)
4. 臺中盆地 (Taichung Basin)
5. 屏東平原 (Pingtung Plain)
6. 蘭陽平原 (Lanyang / Yilan Plain)

輸出：
  data/processed/taiwan_basins_dense.npz (整合六大盆地之多體素封包)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tages.voxels.volume import VoxelGridConfig, VoxelVolumeGenerator

BASIN_CONFIGS = {
    "chianan": {
        "name": "嘉南平原 (Chianan Plain)",
        "desc": "臺灣最大前陸平原，北港基底高區與臺南深前陸槽",
        "lon_min": 120.08, "lon_max": 120.60, "n_lon": 36, # ~1.5km
        "lat_min": 23.15, "lat_max": 23.65, "n_lat": 36,   # ~1.5km
        "depth_min": 0.0, "depth_max": 1000.0, "n_depth": 61 # ~16m
    },
    "zhuoshui": {
        "name": "濁水溪沖積扇 (Zhuoshui Fan)",
        "desc": "臺灣最大現代沖積扇，扇頂礫石至扇端細泥沉積相",
        "lon_min": 120.15, "lon_max": 120.65, "n_lon": 36,
        "lat_min": 23.65, "lat_max": 23.90, "n_lat": 26,
        "depth_min": 0.0, "depth_max": 800.0, "n_depth": 51
    },
    "taipei": {
        "name": "臺北盆地 (Taipei Basin)",
        "desc": "半地塹構造陷落盆地，松山層、景美礫石層與山腳斷層",
        "lon_min": 121.42, "lon_max": 121.62, "n_lon": 31, # ~650m 超高密度
        "lat_min": 24.98, "lat_max": 25.12, "n_lat": 25,   # ~600m 超高密度
        "depth_min": 0.0, "depth_max": 500.0, "n_depth": 51  # 10m 垂直解析度
    },
    "taichung": {
        "name": "臺中盆地 (Taichung Basin)",
        "desc": "車籠埔逆衝斷層下盤，頭嵙山層厚礫岩與盆地沖積層",
        "lon_min": 120.58, "lon_max": 120.78, "n_lon": 31,
        "lat_min": 24.08, "lat_max": 24.28, "n_lat": 26,
        "depth_min": 0.0, "depth_max": 800.0, "n_depth": 51
    },
    "pingtung": {
        "name": "屏東平原 (Pingtung Plain)",
        "desc": "潮州斷層以西厚層砂礫石前陸槽，高屏溪沖積扇體",
        "lon_min": 120.42, "lon_max": 120.68, "n_lon": 31,
        "lat_min": 22.48, "lat_max": 22.78, "n_lat": 31,
        "depth_min": 0.0, "depth_max": 800.0, "n_depth": 51
    },
    "yilan": {
        "name": "蘭陽平原 (Lanyang / Yilan Plain)",
        "desc": "沖繩海槽西延弧後張裂三角洲，扇頂巨礫至海濱細泥",
        "lon_min": 121.68, "lon_max": 121.88, "n_lon": 31,
        "lat_min": 24.58, "lat_max": 24.85, "n_lat": 26,
        "depth_min": 0.0, "depth_max": 600.0, "n_depth": 51
    }
}

def main():
    root = Path(__file__).resolve().parents[1]
    vel_csv = root / "data" / "interim" / "taiwan_velocity_3d.csv"
    
    # 模型載入優先級：特徵蒸餾模型 > 物理引導序列模型 > 增強隨機森林
    model_distilled = root / "models" / "well_seq_net_distilled.pt"
    model_v1 = root / "models" / "well_seq_net_v1.pt"
    model_rf = root / "models" / "tages_rf_augmented.pkl"
    
    if model_distilled.exists():
        model_path = model_distilled
        model_desc = "特徵蒸餾學生神經網路 (DistilledStudentNet - KD + 虛擬電測輔助頭)"
    elif model_v1.exists():
        model_path = model_v1
        model_desc = "時序地質神經網路 (WellboreSeqNet - ResNet + BiLSTM + CRF)"
    else:
        model_path = model_rf
        model_desc = "增強隨機森林基準模型 (Random Forest)"
        
    out_npz = root / "data" / "processed" / "taiwan_basins_dense.npz"
    chianan_single_npz = root / "data" / "processed" / "chianan_3d_voxels_1km.npz"

    print("================================================================")
    print(f"啟動臺灣全島六大代表性盆地與平原高密度 3D 體素模型批次生成")
    print(f"採用地質模型: {model_path.name} [{model_desc}]")
    print("================================================================")

    basin_data_dict = {}

    for b_key, b_info in BASIN_CONFIGS.items():
        print(f"\n正在生成 [{b_info['name']}] 高密度 3D 體素...")
        cfg = VoxelGridConfig(
            lon_min=b_info["lon_min"], lon_max=b_info["lon_max"], n_lon=b_info["n_lon"],
            lat_min=b_info["lat_min"], lat_max=b_info["lat_max"], n_lat=b_info["n_lat"],
            depth_min=b_info["depth_min"], depth_max=b_info["depth_max"], n_depth=b_info["n_depth"]
        )
        gen = VoxelVolumeGenerator(vel_csv, model_path, config=cfg)
        d = gen.generate()
        
        n_pts = d["lithology_3d"].size
        print(f"  成功計算 {n_pts:,} 個體素點 (形狀: {d['lithology_3d'].shape})")
        print(f"  工程基盤平均深度: {np.mean(d['bedrock_engineering_2d']):.1f}m | 聲學基盤平均深度: {np.mean(d['bedrock_acoustic_2d']):.1f}m")
        
        basin_data_dict[f"{b_key}_lats"] = d["lats"]
        basin_data_dict[f"{b_key}_lons"] = d["lons"]
        basin_data_dict[f"{b_key}_depths"] = d["depths"]
        basin_data_dict[f"{b_key}_lith"] = d["lithology_3d"].astype(np.int8)
        basin_data_dict[f"{b_key}_vs"] = d["vs_3d"].astype(np.float32)
        basin_data_dict[f"{b_key}_conf"] = d["confidence_3d"].astype(np.float32)
        basin_data_dict[f"{b_key}_bedrock"] = d["bedrock_depth_2d"].astype(np.float32)
        basin_data_dict[f"{b_key}_bedrock_ac"] = d["bedrock_acoustic_2d"].astype(np.float32)
        basin_data_dict[f"{b_key}_bedrock_eng"] = d["bedrock_engineering_2d"].astype(np.float32)

    # 存為單一六大盆地壓縮檔
    np.savez_compressed(out_npz, **basin_data_dict)
    size_mb = out_npz.stat().st_size / 1024 / 1024
    print(f"\n全台灣六大盆地高密度 3D 體素檔已儲存至: {out_npz} (體積: {size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
