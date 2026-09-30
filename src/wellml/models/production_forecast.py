"""Model 1 — Forecast produksi minyak 30 hari ke depan (data Volve asli).

Pendekatan (dibandingkan head-to-head pada holdout out-of-time):
  * Baseline A : Persistence (rata-rata 30 hari terakhir)
  * Baseline B : Arps Decline Curve Analysis (hyperbolic) per sumur
  * Model utama: LightGBM global (semua sumur) + Optuna tuning dengan
                 walk-forward CV ber-"purge gap" (anti-leakage target overlap)
  * Ketidakpastian: LightGBM quantile regression P10 / P50 / P90
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from wellml.evaluation.metrics import interval_report, pinball, regression_report
from wellml.features.production_features import build_features, feature_columns

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- DCA
def arps_hyperbolic(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


def fit_arps(t: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    m = np.isfinite(q) & (q > 0)
    t, q = t[m], q[m]
    if len(t) < 30:
        return (float(np.nanmedian(q)) if len(q) else 0.0, 1e-4, 0.5)
    p0 = (np.percentile(q[: max(30, len(q) // 10)], 90), 1e-3, 0.5)
    try:
        popt, _ = curve_fit(arps_hyperbolic, t, q, p0=p0, bounds=([0, 1e-6, 0.01], [q.max() * 3, 0.5, 2.0]),
                            loss="soft_l1", f_scale=np.median(q) * 0.2, maxfev=20000)
        return tuple(map(float, popt))
    except Exception:  # pragma: no cover
        return (float(np.median(q)), 1e-4, 0.5)


# ---------------------------------------------------------------- CV
def walk_forward_splits(df: pd.DataFrame, end: pd.Timestamp, n_folds: int, horizon: int):
    """Expanding-window splits. Train hanya memakai baris yang target-nya
    sudah 'selesai' sebelum awal fold validasi (purge = horizon hari)."""
    start = df["date"].min() + pd.Timedelta(days=365 * 2)
    cuts = pd.date_range(start, end, periods=n_folds + 1)
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr = df.index[df["target_date_end"] < a]
        va = df.index[(df["date"] >= a) & (df["date"] < b) & (df["target_date_end"] < end)]
        if len(tr) > 500 and len(va) > 100:
            yield tr, va


# ---------------------------------------------------------------- target transform
# Tree model tidak bisa ekstrapolasi level produksi yang terus menurun. Karena itu
# model memprediksi PERUBAHAN relatif terhadap level 30 hari terakhir:
#   z = log((y + 1) / (oil_mean30 + 1))   ->   y_hat = exp(z_hat) * (oil_mean30 + 1) - 1
# Transformasi monoton, sehingga kuantil tetap valid setelah di-back-transform.
def to_rel(y, level):
    return np.log((np.asarray(y) + 1) / (np.asarray(level) + 1))


def from_rel(z, level):
    return np.clip(np.exp(np.asarray(z)) * (np.asarray(level) + 1) - 1, 0, None)


BASE_PARAMS = dict(n_estimators=600, learning_rate=0.03, verbose=-1, random_state=42)


def _tune(X, y, splits, n_trials: int, seed: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    level = X["oil_mean30"]
    z = to_rel(y, level)
    def objective(trial):
        p = dict(BASE_PARAMS,
                 num_leaves=trial.suggest_int("num_leaves", 8, 64),
                 min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 objective=trial.suggest_categorical("objective", ["regression", "huber", "regression_l1"]))
        errs = []
        for tr, va in splits:
            m = lgb.LGBMRegressor(**p).fit(X.loc[tr], z[X.index.get_indexer(tr)])
            pred = from_rel(m.predict(X.loc[va]), level.loc[va])
            errs.append(np.abs(pred - y.loc[va]).sum() / y.loc[va].abs().sum())
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log.info("Produksi — best CV WAPE %.3f", study.best_value)
    return dict(BASE_PARAMS, subsample_freq=1, **study.best_params), study.best_value


def train_and_evaluate(daily: pd.DataFrame, cfg: dict, n_trials: int = 20, seed: int = 42) -> dict:
    pc = cfg["production"]
    H = pc["horizon_days"]
    feats = build_features(daily, horizon=H, lags=pc["lags"], windows=pc["windows"])
    feats = feats[(feats["days_on_prod"] >= pc["min_history_days"]) & feats["oil_mean30"].notna()]
    feats = feats[feats["target"].notna()].reset_index(drop=True)
    cols = feature_columns(feats)
    test_start = pd.Timestamp(pc["test_start"])

    train_mask = feats["target_date_end"] < test_start
    test_mask = feats["date"] >= test_start
    Xtr, ytr = feats.loc[train_mask, cols], feats.loc[train_mask, "target"]
    Xte, yte = feats.loc[test_mask, cols], feats.loc[test_mask, "target"]

    splits = list(walk_forward_splits(feats[train_mask], test_start, pc["cv_folds"], H))
    params, cv_wape = _tune(Xtr, ytr, splits, n_trials, seed)

    ztr = to_rel(ytr, Xtr["oil_mean30"])
    point = lgb.LGBMRegressor(**params).fit(Xtr, ztr)
    qp = {k: v for k, v in params.items() if k != "objective"}
    qlo, qhi = min(pc["quantiles"]), max(pc["quantiles"])

    # Conformalized Quantile Regression (CQR): kalibrasi lebar interval P10-P90
    # memakai jendela 365 hari terakhir train yang TIDAK dipakai melatih model kuantil.
    cal_start = test_start - pd.Timedelta(days=365)
    fit_m = (feats["target_date_end"] < cal_start) & train_mask
    cal_m = (feats["date"] >= cal_start) & train_mask
    zfit = to_rel(feats.loc[fit_m, "target"], feats.loc[fit_m, "oil_mean30"])
    zcal = to_rel(feats.loc[cal_m, "target"], feats.loc[cal_m, "oil_mean30"])
    lo_m = lgb.LGBMRegressor(**qp, objective="quantile", alpha=qlo).fit(feats.loc[fit_m, cols], zfit)
    hi_m = lgb.LGBMRegressor(**qp, objective="quantile", alpha=qhi).fit(feats.loc[fit_m, cols], zfit)
    Xc = feats.loc[cal_m, cols]
    score = np.maximum(lo_m.predict(Xc) - zcal, zcal - hi_m.predict(Xc))
    nominal = qhi - qlo
    k = min(1.0, np.ceil((len(score) + 1) * nominal) / len(score))
    cqr_q = float(np.quantile(score, k))

    qmodels = {}
    for q in pc["quantiles"]:
        qmodels[q] = lgb.LGBMRegressor(**qp, objective="quantile", alpha=q).fit(Xtr, ztr)

    te = feats.loc[test_mask, ["date", "well", "target", "oil_mean30", "uptime_mean30", "days_on_prod"]].copy()
    lvl = Xte["oil_mean30"]
    te["pred_lgbm"] = from_rel(point.predict(Xte), lvl)
    for q, m in qmodels.items():
        adj = -cqr_q if q == qlo else (cqr_q if q == qhi else 0.0)
        te[f"p{int(q * 100)}"] = from_rel(m.predict(Xte) + adj, lvl)
    # jaga monotonisitas kuantil
    qs = sorted(c for c in te.columns if c.startswith("p") and c[1:].isdigit())
    te[qs] = np.sort(te[qs].to_numpy(), axis=1)
    te["pred_persistence"] = te["oil_mean30"]

    # DCA baseline: fit hanya pada data sebelum test_start
    dca_params = {}
    te["pred_dca"] = np.nan
    for w, g in daily.groupby("well"):
        g = g.sort_values("date").reset_index(drop=True)
        first = g.index[g["oil"] > 0].min()
        g["t"] = g.index - first
        hist = g[(g["date"] < test_start) & (g["t"] >= 0)]
        hist = hist[hist["t"] >= hist["t"].max() - 540]  # regime decline terbaru
        on = (hist["on_stream_hrs"] / 24)
        rate = np.where(on > 0.5, hist["oil"] / on.clip(lower=0.5), np.nan)
        dca_params[w] = fit_arps(hist["t"].to_numpy(float), rate)
        idx = te.index[te["well"] == w]
        if len(idx):
            t0 = te.loc[idx, "days_on_prod"].to_numpy(float)
            steps = np.arange(1, H + 1)
            qf = arps_hyperbolic(t0[:, None] + steps[None, :], *dca_params[w]).mean(axis=1)
            te.loc[idx, "pred_dca"] = qf * te.loc[idx, "uptime_mean30"].fillna(0.8).to_numpy()

    results = {
        "horizon_days": H,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "test_start": str(test_start.date()),
        "cv_wape_best": float(cv_wape), "best_params": params,
        "test": {
            "LightGBM": regression_report(te["target"], te["pred_lgbm"]),
            "Persistence": regression_report(te["target"], te["pred_persistence"]),
            "Arps DCA": regression_report(te["target"], te["pred_dca"]),
        },
        "interval": {**interval_report(te["target"], te["p10"], te["p90"]), "nominal": nominal,
                     "cqr_adjustment_logspace": cqr_q},
        "pinball": {f"q{q}": pinball(te["target"], te[f"p{int(q * 100)}"], q) for q in pc["quantiles"]},
        "per_well": {w: regression_report(g["target"], g["pred_lgbm"]) for w, g in te.groupby("well")},
    }
    return {"results": results, "test_frame": te, "point_model": point, "quantile_models": qmodels,
            "features": cols, "dca_params": dca_params, "cqr_q": cqr_q, "feature_frame": feats, "X_train": Xtr}


def long_term_dca(daily: pd.DataFrame, days: int = 365) -> pd.DataFrame:
    """Forecast jangka panjang per sumur (Arps, fit di seluruh riwayat) + EUR."""
    rows = []
    for w, g in daily.groupby("well"):
        g = g.sort_values("date").reset_index(drop=True)
        first = g.index[g["oil"] > 0].min()
        g["t"] = g.index - first
        g = g[g["t"] >= 0]
        on = g["on_stream_hrs"] / 24
        rate = np.where(on > 0.5, g["oil"] / on.clip(lower=0.5), np.nan)
        # fit pada 540 hari terakhir -> menangkap regime decline terbaru
        tail = g["t"] >= g["t"].max() - 540
        qi, di, b = fit_arps(g.loc[tail, "t"].to_numpy(float), rate[tail.to_numpy()])
        t_last = g["t"].max()
        fut_t = t_last + np.arange(1, days + 1)
        upt = float(on.tail(90).mean())
        fc = arps_hyperbolic(fut_t, qi, di, b) * upt
        rows.append({"well": w, "last_date": g["date"].max(), "cum_oil_sm3": float(g["oil"].sum()),
                     "qi": qi, "di": di, "b": b, "uptime_90d": upt,
                     "next_365d_sm3": float(fc.sum()),
                     "current_rate_sm3d": float(g["oil"].tail(30).mean())})
    return pd.DataFrame(rows)
