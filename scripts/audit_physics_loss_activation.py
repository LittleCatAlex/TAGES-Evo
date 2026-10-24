"""物理引導損失活化率與層析 Vs 特徵診斷 (Physics-Loss Activation & Tomographic-Vs Audit)。

目的：在不重新訓練的前提下，用實際訓練資料檢查 RockPhysicsConsistencyLoss 的四個懲罰項
(a) Vs < vs_gravel_min 且預測礫石、(b) 泊松比 > 門檻且預測粗粒、(c) Vs > vs_clay_max 且預測黏土、
(d) L_invert: z<25 m 且 Vs<180 m/s 且 泊松比>0.38，
是否真的在資料上被觸發，並量化 vs_ms 特徵是否攜帶岩性資訊，或主要是空間位置代理。

所有數字由資料計算，輸出至 reports/phys_loss_activation_audit.json。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """one-way ANOVA effect size：組間變異 / 總變異。"""
    grand = values.mean()
    ss_tot = float(np.sum((values - grand) ** 2))
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "dataset_combined.csv")
    df = df[df["lithology_id"].isin([2, 3, 4, 5, 6, 7])].copy()
    n = len(df)

    vs = df["vs_ms"].values.astype(float)
    nu = df["poisson_ratio"].values.astype(float)
    z = df["depth_m"].values.astype(float)
    lith = df["lithology_id"].values.astype(int)

    # 與 src/tages/models/losses.py 在 is_crustal_scale=True (mean_vs > 1000) 時一致的門檻
    thr = {
        "vs_gravel_min": 1150.0,
        "poisson_gravel_max": 0.370,
        "vs_clay_max": 2500.0,
        "shallow_depth": 25.0,
        "shallow_vs_min": 180.0,
        "shallow_poisson_min": 0.38,
    }
    gate_a = vs < thr["vs_gravel_min"]
    gate_b = nu > thr["poisson_gravel_max"]
    gate_c = vs > thr["vs_clay_max"]
    gate_d = (z < thr["shallow_depth"]) & (vs < thr["shallow_vs_min"]) & (nu > thr["shallow_poisson_min"])

    result = {
        "n_samples": int(n),
        "vs_ms_describe": {k: round(float(v), 1) for k, v in df["vs_ms"].describe().items()},
        "poisson_describe": {k: round(float(v), 4) for k, v in df["poisson_ratio"].describe().items()},
        "gate_activation_fraction": {
            "a_vs_below_gravel_min(1150)": round(float(gate_a.mean()), 4),
            "b_poisson_above_0.37": round(float(gate_b.mean()), 4),
            "c_vs_above_clay_max(2500)": round(float(gate_c.mean()), 4),
            "d_L_invert(z<25,vs<180,nu>0.38)": round(float(gate_d.mean()), 4),
        },
        "gate_activation_count": {
            "a": int(gate_a.sum()), "b": int(gate_b.sum()), "c": int(gate_c.sum()), "d_L_invert": int(gate_d.sum()),
        },
        "vs_by_lithology_mean": {
            str(int(c)): round(float(vs[lith == c].mean()), 1) for c in np.unique(lith)
        },
        "vs_eta_squared_vs_lithology": round(eta_squared(vs, lith), 4),
    }

    # vs 是否主要由「井位 + 深度」決定：以井為群組的 eta^2 與井內變異占比
    wells = df["station_id"].astype(str).values
    result["vs_eta_squared_vs_well"] = round(eta_squared(vs, wells), 4)
    well_mean = df.groupby("station_id")["vs_ms"].transform("mean").values
    within_well_std = float(np.std(vs - well_mean))
    result["vs_within_well_std_ms"] = round(within_well_std, 1)
    result["vs_total_std_ms"] = round(float(vs.std()), 1)

    # 井內 vs 與岩性是否有關：扣掉井均值後再算 eta^2
    result["vs_within_well_eta_squared_vs_lithology"] = round(eta_squared(vs - well_mean, lith), 4)

    out = root / "reports" / "phys_loss_activation_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n已輸出: {out}")


if __name__ == "__main__":
    main()
