"""強震場址效應與 3D 體素地體模型端到端波傳閉環驗證模組 (3D Voxel Model to 1D Haskell-Thomson Wave Propagation Validation)。

本腳本直接從 data/processed/taiwan_basins_dense.npz 提取全臺 32 座代表性 TSMIP 強震觀測站正下方之
3D 體素預測岩性柱與層析波速剖面，代入 Haskell-Thomson 一維多層黏彈性傳遞矩陣計算 3D 體素預測主頻 f0_voxel，
並與 TSMIP 文獻實測主頻 f0_obs 進行外部獨立驗證，同時與四分之一波長粗略公式進行對比。
"""
from run_chianan_ssr_analysis import run_site_effects_validation

if __name__ == "__main__":
    run_site_effects_validation()
