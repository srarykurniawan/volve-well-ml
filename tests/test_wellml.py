"""Unit & integration tests.  Jalankan: pytest -q"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wellml.data.drilling_portfolio import PREDRILL_FEATURES, clean_portfolio, generate
from wellml.economics import emv, npv_success
from wellml.features.production_features import build_features, feature_columns
from wellml.models.portfolio_models import LEAKAGE_COLS, engineer
from wellml.models.production_forecast import arps_hyperbolic, fit_arps, from_rel, to_rel

ROOT = Path(__file__).resolve().parents[1]
ECO = dict(oil_price_usd_bbl=75, sm3_to_bbl=6.2898, opex_usd_bbl=18, royalty_tax_rate=0.5, discount_rate=0.1,
           production_years=10, annual_decline=0.22, sla_penalty_usd_per_day=150000)


def _toy_daily(n=400):
    d = pd.date_range("2010-01-01", periods=n, freq="D")
    oil = 1000 * np.exp(-0.003 * np.arange(n))
    return pd.DataFrame({"date": d, "well": "W1", "oil": oil, "gas": oil * 150, "water": oil * 0.1,
                         "on_stream_hrs": 24.0, **{c: 100.0 for c in
                                                   ["dh_pressure", "dh_temp", "dp_tubing", "choke", "whp", "wht", "dp_choke"]}})


# ---------------- data ----------------
def test_generator_reproducible_and_valid():
    a, b = generate(300, 1), generate(300, 1)
    pd.testing.assert_frame_equal(a, b)
    c = clean_portfolio(a)
    assert (c["water_depth_m"] >= 0).all()
    assert c["net_pay_m"].max() <= 150
    assert set(c["failure"].unique()) <= {0, 1}
    assert ((c["first_year_oil_sm3"] > 0) == (~c["failure_mode"].isin(["dry_hole", "technical"]))).all()


def test_no_target_leakage_in_features():
    X, _ = engineer(clean_portfolio(generate(200, 3)))
    assert not (set(X.columns) & LEAKAGE_COLS)
    assert set(PREDRILL_FEATURES) <= set(X.columns)


def test_categories_are_stable_for_unseen_values():
    df = clean_portfolio(generate(200, 3))
    _, cats = engineer(df)
    new = df.head(3).copy()
    new["formation"] = "UnknownFm"
    X, _ = engineer(new, cats)
    assert X["formation"].isna().all()  # kategori baru -> NaN, bukan crash


# ---------------- production features ----------------
def test_production_target_is_future_only():
    f = build_features(_toy_daily(), horizon=30)
    i = 100
    expected = f["oil"].iloc[i + 1:i + 31].mean()
    assert f["target"].iloc[i] == pytest.approx(expected)
    assert f["target"].iloc[-1:].isna().all()  # tidak ada masa depan -> NaN
    assert "target" not in feature_columns(f) and "oil" not in feature_columns(f)


def test_rel_transform_roundtrip():
    y, lvl = np.array([0, 10, 500.0]), np.array([5, 20, 400.0])
    assert np.allclose(from_rel(to_rel(y, lvl), lvl), y)


def test_arps_fit_recovers_decline():
    t = np.arange(0, 800, dtype=float)
    q = arps_hyperbolic(t, 2000, 0.004, 0.6)
    qi, di, b = fit_arps(t, q * (1 + np.random.default_rng(0).normal(0, 0.02, len(t))))
    assert qi == pytest.approx(2000, rel=0.1)
    assert arps_hyperbolic(1000, qi, di, b) == pytest.approx(arps_hyperbolic(1000, 2000, 0.004, 0.6), rel=0.15)


# ---------------- economics ----------------
def test_npv_monotonic_in_oil_and_emv_penalizes_risk():
    n1, n2 = npv_success([1e5, 5e5], [30, 30], ECO)
    assert n2 > n1
    e_low, e_high = emv([0.1, 0.9], [5e5, 5e5], [30, 30], [30, 30], ECO)
    assert e_low > e_high


# ---------------- trained artefacts (integration) ----------------
@pytest.mark.skipif(not (ROOT / "models" / "portfolio_models.joblib").exists(), reason="jalankan pipeline dulu")
def test_scorer_end_to_end():
    from wellml.predict import WellScorer

    df = clean_portfolio(generate(20, 7))
    out = WellScorer.load(ROOT / "models" / "portfolio_models.joblib").score(df[PREDRILL_FEATURES + ["well_id"]])
    assert len(out) == 20
    assert out["p_fail"].between(0, 1).all() and out["p_sla_breach"].between(0, 1).all()
    assert (out["cost_p10"] <= out["cost_p50"]).all() and (out["cost_p50"] <= out["cost_p90"]).all()
    assert (out["days_p10"] <= out["days_p90"]).all()
    assert out["decision"].notna().all()


@pytest.mark.skipif(not (ROOT / "models" / "portfolio_models.joblib").exists(), reason="jalankan pipeline dulu")
def test_api_score():
    from fastapi.testclient import TestClient

    from api.app import app

    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    payload = [{"area": "North Sea Central", "well_class": "development", "well_profile": "horizontal",
                "formation": "Hugin", "rig_type": "jackup", "water_depth_m": 91, "target_tvd_m": 3100,
                "planned_md_m": 5200, "max_inclination_deg": 89, "pore_pressure_sg": 1.3, "bht_degc": 110,
                "porosity": 0.22, "permeability_md": 300, "net_pay_m": 25, "seismic_quality": 4,
                "offset_wells_5km": 8, "dist_to_producer_km": 1.5, "area_prior_success": 0.55,
                "rig_age_yrs": 12, "crew_experience_yrs": 10, "afe_days": 45, "afe_cost_musd": 32}]
    r = c.post("/score", json=payload)
    assert r.status_code == 200
    assert {"p_fail", "emv_musd", "decision"} <= set(r.json()[0])
    bad = dict(payload[0], porosity=0.9)
    assert c.post("/score", json=[bad]).status_code == 422
