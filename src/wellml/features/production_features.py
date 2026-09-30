"""Feature engineering untuk forecast produksi harian (tanpa leakage).

Semua fitur di hari t hanya memakai informasi <= t. Target = rata-rata volume
minyak harian pada t+1 ... t+H (H = horizon, default 30 hari).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_SENSORS = ["dh_pressure", "dh_temp", "dp_tubing", "choke", "whp", "wht", "dp_choke"]


def build_features(df: pd.DataFrame, horizon: int = 30, lags=(1, 3, 7, 14, 30),
                   windows=(7, 30, 90)) -> pd.DataFrame:
    frames = []
    for w, g in df.sort_values(["well", "date"]).groupby("well"):
        g = g.copy().reset_index(drop=True)
        on = (g["on_stream_hrs"] / 24).clip(0, 1)
        g["uptime"] = on
        g["oil_rate_pd"] = np.where(on > 0.05, g["oil"] / on.clip(lower=0.05), np.nan)  # laju per hari produksi
        liquid = g["oil"] + g["water"]
        g["water_cut"] = np.where(liquid > 0, g["water"] / liquid, np.nan)
        g["gor"] = np.where(g["oil"] > 0, g["gas"] / g["oil"], np.nan)

        first_prod = g.index[g["oil"] > 0].min()
        g["days_on_prod"] = np.clip(np.arange(len(g)) - first_prod, 0, None)
        g["cum_oil"] = g["oil"].cumsum()
        g["cum_water"] = g["water"].cumsum()

        for L in lags:
            g[f"oil_lag{L}"] = g["oil"].shift(L - 1)  # lag1 = nilai hari t
        for W in windows:
            g[f"oil_mean{W}"] = g["oil"].rolling(W, min_periods=max(3, W // 3)).mean()
            g[f"uptime_mean{W}"] = g["uptime"].rolling(W, min_periods=max(3, W // 3)).mean()
            g[f"rate_pd_med{W}"] = g["oil_rate_pd"].rolling(W, min_periods=max(3, W // 3)).median()
        g["oil_std30"] = g["oil"].rolling(30, min_periods=10).std()
        g["trend_7_30"] = g["oil_mean7"] / g["oil_mean30"].replace(0, np.nan)
        g["trend_30_90"] = g["oil_mean30"] / g["oil_mean90"].replace(0, np.nan)
        g["wc_mean30"] = g["water_cut"].rolling(30, min_periods=5).mean()
        g["wc_slope30"] = g["wc_mean30"] - g["wc_mean30"].shift(30)
        g["gor_mean30"] = g["gor"].rolling(30, min_periods=5).mean()
        for c in BASE_SENSORS:
            g[f"{c}_mean7"] = g[c].rolling(7, min_periods=2).mean()
        g["dhp_slope30"] = g["dh_pressure_mean7"] - g["dh_pressure_mean7"].shift(30)
        g["month"] = g["date"].dt.month
        # fitur relatif (scale-free) -> generalisasi antar sumur & antar fase decline
        base = g["oil_mean30"].replace(0, np.nan)
        for L in lags:
            g[f"rel_lag{L}"] = g[f"oil_lag{L}"] / base
        g["rel_mean7"] = g["oil_mean7"] / base
        g["rel_mean90"] = g["oil_mean90"] / base
        g["cv30"] = g["oil_std30"] / base
        g["log_cum_oil"] = np.log1p(g["cum_oil"])

        # target: rata-rata volume harian H hari ke depan (t+1..t+H)
        fwd = g["oil"][::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)
        g["target"] = fwd
        g["target_date_end"] = g["date"] + pd.Timedelta(days=horizon)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["well_cat"] = out["well"].astype("category")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "well", "target", "target_date_end", "oil", "gas", "water", "on_stream_hrs",
               "annulus_press", "water_inj", "flow_kind", "well_type", "oil_rate_pd", "water_cut", "gor",
               *BASE_SENSORS}
    return [c for c in df.columns if c not in exclude]
