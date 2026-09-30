"""Inference: skor prospek sumur baru untuk keputusan go/no-go.

    from wellml.predict import WellScorer
    scorer = WellScorer.load()
    scorer.score(pd.DataFrame([prospect_dict]))
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from wellml.config import load_config
from wellml.economics import npv_success
from wellml.models.portfolio_models import engineer


class WellScorer:
    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WellScorer":
        path = path or Path(load_config()["paths"]["models_dir"]) / "portfolio_models.joblib"
        return cls(joblib.load(path))

    def _q(self, key: str, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, c = self.b[key]["quantiles"], self.b[key]["cqr"]
        p10 = np.exp(m[0.1].predict(X) - c)
        p50 = np.exp(m[0.5].predict(X))
        p90 = np.exp(m[0.9].predict(X) + c)
        return tuple(np.sort(np.vstack([p10, p50, p90]), axis=0))

    def score(self, prospects: pd.DataFrame) -> pd.DataFrame:
        df = prospects.copy()
        sla = pd.to_numeric(df["sla_days"], errors="coerce") if "sla_days" in df else pd.Series(np.nan, index=df.index)
        df["sla_days"] = sla.fillna(df["afe_days"].astype(float) * 1.10)
        X, _ = engineer(df, self.b["categories"])
        X = X[self.b["features"]]
        eco = self.b["economics"]
        out = pd.DataFrame(index=df.index)
        if "well_id" in df:
            out["well_id"] = df["well_id"]
        out["p_fail"] = self.b["failure"]["model"].predict_proba(X)[:, 1]
        out["p_sla_breach"] = self.b["sla"]["model"].predict_proba(X)[:, 1]
        out["cost_p10"], out["cost_p50"], out["cost_p90"] = self._q("cost", X)
        out["days_p10"], out["days_p50"], out["days_p90"] = self._q("duration", X)
        out["oil_p10"], out["oil_p50"], out["oil_p90"] = self._q("oil", X)
        out["npv_if_success_musd"] = npv_success(out["oil_p50"], out["cost_p50"], eco)
        overrun = np.clip(out["days_p50"] - df["sla_days"], 0, None)
        out["expected_sla_penalty_musd"] = out["p_sla_breach"] * overrun * eco["sla_penalty_usd_per_day"] / 1e6
        out["emv_musd"] = ((1 - out["p_fail"]) * out["npv_if_success_musd"] - out["p_fail"] * out["cost_p50"]
                           - out["expected_sla_penalty_musd"])
        thr = self.b["failure"]["threshold"]
        out["decision"] = np.select(
            [out["emv_musd"] <= 0, out["p_fail"] >= thr, out["p_sla_breach"] >= 0.6],
            ["NO-GO (EMV negatif)", "REVIEW (risiko gagal tinggi)", "GO + mitigasi SLA"], "GO")
        return out.round(4)
