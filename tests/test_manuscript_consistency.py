"""Automated consistency and integrity tests for the TAGES manuscript and verified datasets."""
from pathlib import Path
import json
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "國際科展作品說明書_最新進度版_EAEV.md"
EGDT_CSV_PATH = ROOT / "data" / "raw" / "seismic" / "tsmip_egdt_site_truth.csv"
SITE_TRUTH_CSV_PATH = ROOT / "data" / "raw" / "seismic" / "tsmip_site_truth.csv"
TSMIP_SUMMARY_PATH = ROOT / "reports" / "tsmip_closed_loop_summary.csv"
TSMIP_METRICS_PATH = ROOT / "reports" / "tsmip_closed_loop_metrics.json"
SR_METRICS_PATH = ROOT / "reports" / "site_response_metrics.json"
PHYS_AUDIT_PATH = ROOT / "reports" / "phys_loss_activation_audit.json"
DEEP_METRICS_PATH = ROOT / "reports" / "seismic_site_effects_deep_metrics.json"


def test_egdt_site_truth_provenance():
    """Verify that all 32 EGDT strong-motion stations have verified provenance and no bogus earthquake_event column."""
    for path in (EGDT_CSV_PATH, SITE_TRUTH_CSV_PATH):
        assert path.exists(), f"Missing {path}"
        df = pd.read_csv(path)
        assert len(df) == 32, f"Expected 32 stations in {path.name}, got {len(df)}"
        assert "earthquake_event" not in df.columns, f"Bogus earthquake_event column still in {path.name}"
        assert "source_verified" in df.columns, f"Missing source_verified in {path.name}"
        assert "source_ref" in df.columns, f"Missing source_ref in {path.name}"
        assert (df["source_verified"] == "yes").all(), f"All rows must have source_verified == 'yes' in {path.name}"
        assert df["source_ref"].notna().all() and (df["source_ref"].str.len() > 10).all()


def test_manuscript_no_stale_or_corrupted_strings():
    """Ensure no stale metrics, unverified station mis-mappings, or corrupted LaTeX strings remain in the manuscript."""
    text = DOC_PATH.read_text(encoding="utf-8")

    forbidden_substrings = [
        "0.8285",
        "0.6570",
        "0.350\\text{ Hz}",
        "0.350 Hz",
        "0.7981",
        "0.5557",
        "0.286\\text{ Hz}",
        "109.7\\text{ m}",
        "0.9082",
        "\text{RMSE}",  # literal tab + ext{RMSE}
        "12 TSMIP stations",
        "國震中心公開工程鑽探報告",
        "100% 盲測，不使用測站實測",
        "Lee, S.-J., Liang, W.-T.",
        "布袋 (Budai)",  # old mis-mapped CHY010 name
    ]
    for bad in forbidden_substrings:
        assert bad not in text, f"Found forbidden/stale string in manuscript: {bad!r}"


def test_manuscript_matches_verified_json_metrics():
    """Verify key numbers in the manuscript match the exact JSON/CSV outputs from scripts."""
    text = DOC_PATH.read_text(encoding="utf-8")
    tsmip_metrics = json.loads(TSMIP_METRICS_PATH.read_text(encoding="utf-8"))
    sr_metrics = json.loads(SR_METRICS_PATH.read_text(encoding="utf-8"))
    phys_audit = json.loads(PHYS_AUDIT_PATH.read_text(encoding="utf-8"))

    # Check TSMIP closed-loop metrics
    assert f"{tsmip_metrics['f0_voxel_vs_obs']['pearson_r']:.4f}" in text
    assert f"{tsmip_metrics['f0_voxel_vs_obs']['spearman_rho']:.4f}" in text
    assert f"{tsmip_metrics['f0_voxel_vs_obs']['r2']:.4f}" in text
    assert f"{tsmip_metrics['f0_voxel_vs_obs']['rmse_hz']:.3f}" in text
    assert f"{tsmip_metrics['quarter_wavelength_vs_obs']['rmse_hz']:.3f}" in text
    assert f"{tsmip_metrics['acoustic_bedrock_vs_lit_depth']['pearson_r']:.4f}" in text
    assert f"{tsmip_metrics['acoustic_bedrock_vs_lit_depth']['rmse_m']:.1f}" in text

    # Check Site Response metrics
    clay = sr_metrics["facies_theoretical_models"]["clay_soft"]
    sand = sr_metrics["facies_theoretical_models"]["sand_medium"]
    gravel = sr_metrics["facies_theoretical_models"]["gravel_stiff"]
    assert f"{clay['impedance_ratio_alpha']:.3f}" in text
    assert f"{sand['impedance_ratio_alpha']:.3f}" in text
    assert f"{gravel['impedance_ratio_alpha']:.3f}" in text
    assert f"{clay['rs_peak_sa_pga']:.2f}" in text
    assert f"{sand['rs_peak_sa_pga']:.2f}" in text
    assert f"{gravel['rs_peak_sa_pga']:.2f}" in text

    # Check Physics Loss activation audit metrics
    assert f"{phys_audit['vs_ms_describe']['min']:.1f}" in text
    assert f"{phys_audit['vs_ms_describe']['max']:.1f}" in text
    assert f"{phys_audit['gate_activation_fraction']['c_vs_above_clay_max(2500)'] * 100:.2f}%" in text


def test_english_abstract_word_count():
    """Verify the English abstract strictly complies with the TISF/ISEF <= 250 words limit."""
    text = DOC_PATH.read_text(encoding="utf-8")
    start = text.index("### 英文摘要（Abstract in English, ≤250 words）")
    end = text.index("**Keywords**:", start)
    eng_block = text[start:end].splitlines()[1:]  # skip heading line
    eng_text = " ".join(line.strip() for line in eng_block if line.strip())
    words = re.findall(r"\b\w+(?:[-']\w+)*\b", eng_text)
    assert len(words) <= 250, f"English abstract word count {len(words)} exceeds 250 words limit"


def test_manuscript_seismic_site_effects_deep_metrics():
    """Verify deepened strong-motion site effects, basin-edge regression, and Meinong observations."""
    text = DOC_PATH.read_text(encoding="utf-8")
    assert DEEP_METRICS_PATH.exists(), f"Missing {DEEP_METRICS_PATH}"
    deep_metrics = json.loads(DEEP_METRICS_PATH.read_text(encoding="utf-8"))

    edge_reg = deep_metrics["basin_edge_effect"]["linear_regression"]
    near_edge = deep_metrics["basin_edge_effect"]["near_edge_stations_d_lt_6km"]
    basin_center = deep_metrics["basin_edge_effect"]["basin_center_stations_d_ge_6km"]

    # Regression metrics
    assert f"{edge_reg['slope']:.5f}" in text
    assert f"{near_edge['mean_bias_hz']:.3f}" in text
    assert f"{basin_center['mean_bias_hz']:.3f}" in text

    # Figure 3-13 reference
    assert "reports/seismic_site_effects_deep_analysis.png" in text
    assert "圖 3-13" in text

    # 2016 Meinong earthquake real stations and observations
    assert "ALS" in text
    assert "CHY015" in text
    assert "CHY054" in text
    assert "81.3" in text or "81.34" in text
    assert "262.4" in text or "262.36" in text
    assert "152.4" in text

    # Objective Weiguan collapse factors
    assert "軟弱底層" in text
    assert "135" in text
    assert "90" in text

    # Building code and NCREE-17-004 discussion
    assert "NCREE-17-004" in text
    assert "微區一" in text

