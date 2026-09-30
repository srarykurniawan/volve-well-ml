"""Pipeline end-to-end: ingest -> clean -> features -> tuning -> training ->
evaluasi -> explainability -> artefak model + laporan + data dashboard.

    python -m wellml.pipeline                 # jalankan semua
    python -m wellml.pipeline --fast          # trial Optuna lebih sedikit (CI / demo)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wellml import reporting
from wellml.config import load_config
from wellml.data import drilling_portfolio as dp
from wellml.data import ingest_volve
from wellml.models import portfolio_models, production_forecast

warnings.filterwarnings("ignore")
log = logging.getLogger("wellml")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(type(o))


def load_portfolio(cfg) -> pd.DataFrame:
    p = Path(cfg["paths"]["portfolio_csv"])
    if p.exists():
        log.info("Memakai portofolio dari %s", p)
        df = pd.read_csv(p)
    else:
        log.info("Membangkitkan portofolio augmented (n=%d)", cfg["portfolio"]["n_wells"])
        df = dp.generate(cfg["portfolio"]["n_wells"], cfg["seed"])
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return dp.clean_portfolio(df)


def run(fast: bool = False, config_path: str | None = None) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    if fast:
        cfg["portfolio"]["optuna_trials"] = 5
    models_dir, rep = Path(cfg["paths"]["models_dir"]), Path(cfg["paths"]["reports_dir"])
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Produksi (Volve) ----------------
    log.info("[1/3] Ingest & cleaning Volve")
    daily = ingest_volve.run(cfg["paths"]["raw_volve"], cfg["paths"]["processed_dir"])
    log.info("[1/3] Training forecast produksi")
    prod = production_forecast.train_and_evaluate(daily, cfg, n_trials=6 if fast else 25, seed=cfg["seed"])
    lt = production_forecast.long_term_dca(daily)
    joblib.dump({"point": prod["point_model"], "quantiles": prod["quantile_models"], "features": prod["features"],
                 "cqr_q": prod["cqr_q"], "horizon": cfg["production"]["horizon_days"]},
                models_dir / "production_forecast.joblib")
    prod["test_frame"].to_csv(rep / "production_test_predictions.csv", index=False)
    lt.to_csv(rep / "production_long_term_dca.csv", index=False)
    reporting.production_forecast(prod["test_frame"], rep / "figures" / "production_forecast.png")
    prod["results"]["shap"] = portfolio_models.shap_importance(prod["point_model"], prod["X_train"])

    # ---------------- 2. Portofolio pengeboran ----------------
    log.info("[2/3] Training model risiko, biaya, durasi, SLA, produksi pra-bor")
    pf = load_portfolio(cfg)
    port = portfolio_models.train_all(pf, cfg, seed=cfg["seed"])
    bundle = {
        "categories": port["categories"], "features": port["features"],
        "failure": {"model": port["failure"]["model"], "threshold": port["failure"]["threshold"]},
        "sla": {"model": port["sla"]["model"], "threshold": port["sla"]["threshold"]},
        **{k: {"quantiles": port[k]["quantile_models"], "cqr": port[k]["cqr"]} for k in ("cost", "duration", "oil")},
        "economics": cfg["economics"],
    }
    joblib.dump(bundle, models_dir / "portfolio_models.joblib")
    port["test_frame"].to_csv(rep / "portfolio_test_predictions.csv", index=False)
    tf = port["test_frame"]
    reporting.calibration(port["failure"]["results"]["calibration"], rep / "figures" / "failure_calibration.png",
                          "Kalibrasi probabilitas gagal (test)")
    reporting.pred_vs_actual(tf["actual_cost_musd"], tf["cost_p50"], tf["afe_cost_musd"],
                             rep / "figures" / "cost_pred_vs_actual.png", "Biaya sumur: ML vs AFE", "juta USD")
    reporting.pred_vs_actual(tf["actual_days"], tf["days_p50"], tf["afe_days"],
                             rep / "figures" / "duration_pred_vs_actual.png", "Durasi: ML vs AFE", "hari")
    reporting.shap_bar(port["failure"]["results"]["shap"], rep / "figures" / "failure_shap.png",
                       "Driver risiko kegagalan (SHAP)")

    # ---------------- 3. Metrik & data dashboard ----------------
    log.info("[3/3] Menyimpan metrik & data dashboard")
    metrics = {
        "production": prod["results"],
        "portfolio": {k: port[k]["results"] for k in ("failure", "sla", "cost", "duration", "oil")},
        "business": port["business"],
        "data": {"volve_rows_clean": int(len(daily)), "volve_wells": sorted(daily["well"].unique().tolist()),
                 "portfolio_rows": int(len(pf)), "portfolio_train": port["n_train"], "portfolio_test": port["n_test"],
                 "portfolio_source": str(pf["data_source"].iloc[0]) if "data_source" in pf else "user"},
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(rep / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # ringkas untuk dashboard (seri waktu mingguan agar ringan)
    te = prod["test_frame"].copy()
    hist = daily.groupby(["well", pd.Grouper(key="date", freq="W")])["oil"].mean().reset_index()
    dash = {
        "metrics": metrics,
        "production_history": hist.assign(date=hist["date"].dt.strftime("%Y-%m-%d")).round(1).to_dict("records"),
        "production_test": te.assign(date=te["date"].dt.strftime("%Y-%m-%d"))[
            ["date", "well", "target", "pred_lgbm", "p10", "p90", "pred_persistence", "pred_dca"]].round(1).to_dict("records"),
        "long_term": lt.assign(last_date=lt["last_date"].dt.strftime("%Y-%m-%d")).round(4).to_dict("records"),
        "portfolio_test": tf.round(4).to_dict("records"),
    }
    with open(rep / "dashboard_data.json", "w") as f:
        json.dump(dash, f, default=_json_default)
    log.info("Selesai dalam %.1f detik. Artefak: %s, %s", time.time() - t0, models_dir, rep)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(fast=a.fast, config_path=a.config)
