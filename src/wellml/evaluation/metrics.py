"""Metrik evaluasi regresi, klasifikasi & kuantil."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)


def regression_report(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    nz = y > 0
    return {
        "MAE": float(mean_absolute_error(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "WAPE": float(np.abs(y - yhat).sum() / np.abs(y).sum()),
        "MAPE": float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz]))),
        "R2": float(r2_score(y, yhat)),
        "bias_pct": float((yhat.sum() - y.sum()) / y.sum()),
        "n": int(len(y)),
    }


def interval_report(y, lo, hi) -> dict:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return {"coverage_P10_P90": float(np.mean((y >= lo) & (y <= hi))),
            "mean_width": float(np.mean(hi - lo))}


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def classification_report(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= threshold).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "F1": float(f1_score(y, yhat, zero_division=0)),
        "threshold": float(threshold),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "pred": float(p[m].mean()), "obs": float(y[m].mean()),
                         "n": int(m.sum())})
    return rows
