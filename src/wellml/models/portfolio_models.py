"""Model 2-5 — risiko kegagalan, biaya, durasi, SLA & produksi pra-bor.

Semua model hanya memakai fitur PRA-PENGEBORAN (tersedia saat keputusan
go/no-go), divalidasi out-of-time (latih <= 2021, uji >= 2022) dan
dibandingkan dengan baseline yang dipakai perusahaan saat ini (AFE).
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wellml.data.drilling_portfolio import PREDRILL_FEATURES
from wellml.economics import npv_success
from wellml.evaluation.metrics import (calibration_table, classification_report, interval_report,
                                       regression_report)

log = logging.getLogger(__name__)

CAT_COLS = ["area", "well_class", "well_profile", "formation", "rig_type", "mud_type"]
LEAKAGE_COLS = {"failure", "failure_mode", "actual_days", "actual_cost_musd", "sla_breach",
                "first_year_oil_sm3", "data_source", "well_id"}


# ------------------------------------------------------------------ features
def engineer(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    X = df[[c for c in PREDRILL_FEATURES if c in df]].copy()
    for c in X.columns:
        if c not in CAT_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    assert not (set(X.columns) & LEAKAGE_COLS), "Leakage: kolom target masuk fitur!"
    X["log_kh"] = np.log1p(X["permeability_md"] * X["net_pay_m"])
    X["md_tvd_ratio"] = X["planned_md_m"] / X["target_tvd_m"]
    X["afe_cost_per_day"] = X["afe_cost_musd"] / X["afe_days"]
    X["afe_m_per_day"] = X["planned_md_m"] / X["afe_days"]
    X["pp_x_depth"] = X["pore_pressure_sg"] * X["target_tvd_m"] / 1000
    X["exploratory_risk"] = (X["well_class"] != "development").astype(int) / np.log1p(X["offset_wells_5km"] + 1)
    categories = categories or {c: sorted(X[c].dropna().unique().tolist()) for c in CAT_COLS}
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].where(X[c].isin(categories[c])), categories=categories[c])
    return X, categories


# ------------------------------------------------------------------ tuning
def tune_lgbm(X, y, kind: str, n_trials: int, folds: int, seed: int) -> tuple[dict, float]:
    import optuna  # import lokal: tidak dibutuhkan saat inference/dashboard

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = (StratifiedKFold if kind == "binary" else KFold)(folds, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))

    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 150, 900, step=50),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 7, 63),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                 random_state=seed, verbose=-1)
        losses = []
        for tr, va in splits:
            if kind == "binary":
                m = lgb.LGBMClassifier(**p).fit(X.iloc[tr], y.iloc[tr])
                pr = np.clip(m.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
                yv = y.iloc[va].to_numpy()
                losses.append(-np.mean(yv * np.log(pr) + (1 - yv) * np.log(1 - pr)))
            else:
                m = lgb.LGBMRegressor(**p).fit(X.iloc[tr], y.iloc[tr])
                losses.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va])))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params, subsample_freq=1, random_state=seed, verbose=-1)
    return params, float(study.best_value)


def shap_importance(model, X: pd.DataFrame, top: int = 15) -> list[dict]:
    import shap  # import lokal: hanya dipakai saat training

    sample = X.sample(min(800, len(X)), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:top]
    # arah pengaruh: korelasi nilai fitur (numerik) dengan SHAP
    out = []
    for i in order:
        col = sample.columns[i]
        vals = sample[col]
        direction = None
        if pd.api.types.is_numeric_dtype(vals) and vals.notna().sum() > 10 and vals.nunique() > 1:
            ok = vals.notna().to_numpy()
            direction = float(np.corrcoef(vals.to_numpy()[ok], sv[ok, i])[0, 1])
        out.append({"feature": col, "mean_abs_shap": float(imp[i]), "direction": direction})
    return out


def _cats_to_object(X):
    X = X.copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(object)
    return X


def make_logreg(columns) -> "Pipeline":
    from sklearn.preprocessing import FunctionTransformer

    num = [c for c in columns if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CAT_COLS),
    ])
    return make_pipeline(FunctionTransformer(_cats_to_object), pre, LogisticRegression(max_iter=3000, C=0.5))


class BlendClassifier:
    """Rata-rata probabilitas beberapa model terkalibrasi (soft-voting)."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------ classifier
def fit_classifier(name, Xtr, ytr, Xte, yte, cfg, seed, value_fn=None):
    pc = cfg["portfolio"]
    params, cv_loss = tune_lgbm(Xtr, ytr, "binary", pc["optuna_trials"], pc["cv_folds"], seed)
    raw = lgb.LGBMClassifier(**params).fit(Xtr, ytr)
    skf = StratifiedKFold(pc["cv_folds"], shuffle=True, random_state=seed)
    lgb_cal = CalibratedClassifierCV(lgb.LGBMClassifier(**params), method="isotonic", cv=3)
    oof_lgb = cross_val_predict(lgb_cal, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    oof_lr = cross_val_predict(make_logreg(Xtr.columns), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    oof_blend = (oof_lgb + oof_lr) / 2
    cands = {"LightGBM (calibrated)": oof_lgb, "Logistic Regression": oof_lr, "Blend (LGBM+LR)": oof_blend}
    cv_scores = {k: _logloss(ytr, v) for k, v in cands.items()}
    champion = min(cv_scores, key=cv_scores.get)
    oof = cands[champion]

    m_lgb = CalibratedClassifierCV(lgb.LGBMClassifier(**params), method="isotonic", cv=pc["cv_folds"]).fit(Xtr, ytr)
    m_lr = make_logreg(Xtr.columns).fit(Xtr, ytr)
    fitted = {"LightGBM (calibrated)": m_lgb, "Logistic Regression": m_lr,
              "Blend (LGBM+LR)": BlendClassifier([m_lgb, m_lr])}
    cal = fitted[champion]
    # threshold: maksimalkan nilai bisnis (jika diberikan) atau F1 pada OOF
    grid = np.linspace(0.1, 0.9, 81)
    if value_fn is not None:
        scores = [value_fn(oof, t) for t in grid]
    else:
        scores = [classification_report(ytr, oof, t)["F1"] for t in grid]
    thr = float(grid[int(np.argmax(scores))])

    p_te = cal.predict_proba(Xte)[:, 1]
    test = {k: classification_report(yte, m.predict_proba(Xte)[:, 1], thr) for k, m in fitted.items()}
    test["LightGBM (raw)"] = classification_report(yte, raw.predict_proba(Xte)[:, 1], thr)
    res = {
        "cv_logloss_lgbm_tuning": cv_loss, "cv_logloss_candidates": cv_scores, "champion": champion,
        "threshold": thr, "params": params, "test": test,
        "calibration": calibration_table(yte, p_te),
        "shap": shap_importance(raw, Xte),
    }
    log.info("%s — champion=%s | test AUC %.3f, Brier %.3f", name, champion, test[champion]["ROC_AUC"],
             test[champion]["Brier"])
    return {"model": cal, "raw_model": raw, "threshold": thr, "results": res, "p_test": p_te}


# ------------------------------------------------------------------ regressor (log-target + CQR)
def fit_regressor(name, Xtr, ytr, Xte, yte, years_tr, cfg, seed, baseline_te=None, baseline_name="AFE"):
    pc = cfg["portfolio"]
    ztr = np.log(ytr)
    params, cv_mae = tune_lgbm(Xtr, ztr, "regression", pc["optuna_trials"], pc["cv_folds"], seed)
    point = lgb.LGBMRegressor(**params).fit(Xtr, ztr)
    qp = {k: v for k, v in params.items()}
    q = {a: lgb.LGBMRegressor(**qp, objective="quantile", alpha=a).fit(Xtr, ztr) for a in (0.1, 0.5, 0.9)}

    # CQR out-of-time: fit <= (tahun_max-2), kalibrasi 2 tahun terakhir train
    cut = years_tr.max() - 2
    f, c = years_tr <= cut, years_tr > cut
    lo = lgb.LGBMRegressor(**qp, objective="quantile", alpha=0.1).fit(Xtr[f], ztr[f])
    hi = lgb.LGBMRegressor(**qp, objective="quantile", alpha=0.9).fit(Xtr[f], ztr[f])
    s = np.maximum(lo.predict(Xtr[c]) - ztr[c], ztr[c] - hi.predict(Xtr[c]))
    cqr = float(np.quantile(s, min(1.0, np.ceil((len(s) + 1) * 0.8) / len(s))))

    pred = np.exp(point.predict(Xte))
    p10 = np.exp(q[0.1].predict(Xte) - cqr)
    p50 = np.exp(q[0.5].predict(Xte))
    p90 = np.exp(q[0.9].predict(Xte) + cqr)
    p10, p50, p90 = np.sort(np.vstack([p10, p50, p90]), axis=0)
    res = {"cv_mae_log": cv_mae, "params": params, "cqr_adjustment_logspace": cqr,
           "test": {"LightGBM": regression_report(yte, pred)},
           "interval": interval_report(yte, p10, p90),
           "shap": shap_importance(point, Xte)}
    if baseline_te is not None:
        res["test"][baseline_name] = regression_report(yte, baseline_te)
    log.info("%s — test MAPE %.3f vs %s %.3f", name, res["test"]["LightGBM"]["MAPE"], baseline_name,
             res["test"].get(baseline_name, {}).get("MAPE", np.nan))
    return {"model": point, "quantile_models": q, "cqr": cqr, "results": res,
            "pred_test": pred, "p10": p10, "p50": p50, "p90": p90}


# ------------------------------------------------------------------ orchestration
def train_all(df: pd.DataFrame, cfg: dict, seed: int = 42) -> dict:
    eco = cfg["economics"]
    tr = df[df["spud_year"] < cfg["portfolio"]["test_from_year"]].reset_index(drop=True)
    te = df[df["spud_year"] >= cfg["portfolio"]["test_from_year"]].reset_index(drop=True)
    Xtr, cats = engineer(tr)
    Xte, _ = engineer(te, cats)
    out = {"categories": cats, "features": list(Xtr.columns), "n_train": len(tr), "n_test": len(te)}

    # nilai bisnis sebuah threshold: bor hanya jika p_fail < thr
    npv_tr = npv_success(tr["first_year_oil_sm3"], tr["actual_cost_musd"], eco)
    realized = np.where(tr["failure"] == 1, -tr["actual_cost_musd"], npv_tr)

    def value_fn(p, thr):
        return float(realized[p < thr].sum())

    out["failure"] = fit_classifier("Failure", Xtr, tr["failure"], Xte, te["failure"], cfg, seed, value_fn)
    out["sla"] = fit_classifier("SLA breach", Xtr, tr["sla_breach"], Xte, te["sla_breach"], cfg, seed)
    out["cost"] = fit_regressor("Cost", Xtr, tr["actual_cost_musd"], Xte, te["actual_cost_musd"],
                                tr["spud_year"], cfg, seed, te["afe_cost_musd"])
    out["duration"] = fit_regressor("Duration", Xtr, tr["actual_days"], Xte, te["actual_days"],
                                    tr["spud_year"], cfg, seed, te["afe_days"])
    ok_tr, ok_te = tr["first_year_oil_sm3"] > 0, te["first_year_oil_sm3"] > 0
    grp = tr[ok_tr].groupby(["area", "well_profile"])["first_year_oil_sm3"].median()
    base_oil = te[ok_te].apply(lambda r: grp.get((r["area"], r["well_profile"]),
                                                 tr.loc[ok_tr, "first_year_oil_sm3"].median()), axis=1)
    out["oil"] = fit_regressor("First-year oil", Xtr[ok_tr.to_numpy()], tr.loc[ok_tr, "first_year_oil_sm3"],
                               Xte[ok_te.to_numpy()], te.loc[ok_te, "first_year_oil_sm3"],
                               tr.loc[ok_tr, "spud_year"], cfg, seed, base_oil.to_numpy(), "Analog median (area x profile)")

    # ---------- evaluasi bisnis pada test set ----------
    pf = out["failure"]["p_test"]
    oil_all = np.exp(out["oil"]["quantile_models"][0.5].predict(Xte))
    cost_all = out["cost"]["pred_test"]
    npv_pred = npv_success(oil_all, cost_all, eco)
    emv_pred = (1 - pf) * npv_pred - pf * cost_all
    npv_te = npv_success(te["first_year_oil_sm3"], te["actual_cost_musd"], eco)
    realized_te = np.where(te["failure"] == 1, -te["actual_cost_musd"], npv_te)
    drill_all = realized_te.sum()
    drill_model = realized_te[emv_pred > 0].sum()
    thr = out["failure"]["threshold"]
    out["business"] = {
        "n_prospects_test": int(len(te)),
        "baseline_drill_all": {"n_drilled": int(len(te)), "n_failed": int(te["failure"].sum()),
                               "failure_rate": float(te["failure"].mean()),
                               "realized_value_musd": float(drill_all),
                               "capex_musd": float(te["actual_cost_musd"].sum())},
        "model_emv_gt_0": {"n_drilled": int((emv_pred > 0).sum()),
                           "n_failed": int(te.loc[emv_pred > 0, "failure"].sum()),
                           "failure_rate": float(te.loc[emv_pred > 0, "failure"].mean()),
                           "realized_value_musd": float(drill_model),
                           "capex_musd": float(te.loc[emv_pred > 0, "actual_cost_musd"].sum())},
        "failure_threshold": thr,
        "afe_cost_error_musd_total": float(np.abs(te["actual_cost_musd"] - te["afe_cost_musd"]).sum()),
        "model_cost_error_musd_total": float(np.abs(te["actual_cost_musd"] - cost_all).sum()),
        "sla_breach_rate_test": float(te["sla_breach"].mean()),
    }
    te_out = te[["well_id", "spud_year", "area", "well_class", "well_profile", "rig_type", "afe_days",
                 "sla_days", "afe_cost_musd", "actual_days", "actual_cost_musd", "failure", "failure_mode",
                 "sla_breach", "first_year_oil_sm3"]].copy()
    te_out["p_fail"] = pf
    te_out["p_sla_breach"] = out["sla"]["p_test"]
    te_out["cost_p10"], te_out["cost_p50"], te_out["cost_p90"] = out["cost"]["p10"], out["cost"]["p50"], out["cost"]["p90"]
    te_out["days_p10"], te_out["days_p50"], te_out["days_p90"] = out["duration"]["p10"], out["duration"]["p50"], out["duration"]["p90"]
    te_out["oil_p50_if_success"] = oil_all
    te_out["emv_musd"] = emv_pred
    te_out["decision"] = np.where(emv_pred > 0, "DRILL", "REVIEW / DEFER")
    out["test_frame"] = te_out
    return out
