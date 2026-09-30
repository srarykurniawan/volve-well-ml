"""Ingest & cleaning data produksi harian Volve (Equinor Open Data).

Sumber : https://www.equinor.com/energy/volve-data-sharing
Isi     : 15.634 baris produksi harian, 7 wellbore, 2007-09 s/d 2016-12.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RENAME = {
    "DATEPRD": "date",
    "NPD_WELL_BORE_NAME": "well",
    "ON_STREAM_HRS": "on_stream_hrs",
    "AVG_DOWNHOLE_PRESSURE": "dh_pressure",
    "AVG_DOWNHOLE_TEMPERATURE": "dh_temp",
    "AVG_DP_TUBING": "dp_tubing",
    "AVG_ANNULUS_PRESS": "annulus_press",
    "AVG_CHOKE_SIZE_P": "choke",
    "AVG_WHP_P": "whp",
    "AVG_WHT_P": "wht",
    "DP_CHOKE_SIZE": "dp_choke",
    "BORE_OIL_VOL": "oil",
    "BORE_GAS_VOL": "gas",
    "BORE_WAT_VOL": "water",
    "BORE_WI_VOL": "water_inj",
    "FLOW_KIND": "flow_kind",
    "WELL_TYPE": "well_type",
}
SENSOR_COLS = ["dh_pressure", "dh_temp", "dp_tubing", "annulus_press", "choke", "whp", "wht", "dp_choke"]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Daily Production Data")
    df = df.rename(columns=RENAME)[list(RENAME.values())]
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning yang robust & dapat diaudit.

    1. Hanya baris produksi (injeksi dipisah).
    2. Buang sumur produsen dengan riwayat produksi < 180 hari (F-5 = injektor yang sempat produksi).
    3. Nilai sensor 0 saat sumur shut-in = missing, bukan pengukuran.
    4. Volume negatif -> 0; outlier ekstrem sensor di-clip (IQR x 5 per sumur).
    5. Kalender harian lengkap per sumur (hari hilang = shut-in, oil = 0).
    6. Sensor diisi forward-fill terbatas (maks 7 hari) per sumur.
    """
    n0 = len(df)
    df = df[df["flow_kind"] == "production"].copy()
    df = df.sort_values(["well", "date"]).drop_duplicates(["well", "date"])

    prod_days = df[df["oil"] > 0].groupby("well").size()
    keep = prod_days[prod_days >= 180].index
    df = df[df["well"].isin(keep)]

    for c in ["oil", "gas", "water", "on_stream_hrs"]:
        df[c] = df[c].fillna(0).clip(lower=0)
    df["on_stream_hrs"] = df["on_stream_hrs"].clip(upper=24)

    shut = df["on_stream_hrs"] <= 0
    for c in SENSOR_COLS:
        df.loc[shut | (df[c] <= 0), c] = np.nan

    def _clip(g: pd.DataFrame) -> pd.DataFrame:
        for c in SENSOR_COLS:
            q1, q3 = g[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                g[c] = g[c].clip(q1 - 5 * iqr, q3 + 5 * iqr)
        return g

    df = df.groupby("well", group_keys=False)[df.columns].apply(_clip)

    # kalender lengkap
    frames = []
    for w, g in df.groupby("well"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g.index.name = "date"
        g["well"] = w
        for c in ["oil", "gas", "water", "on_stream_hrs"]:
            g[c] = g[c].fillna(0)
        g[SENSOR_COLS] = g[SENSOR_COLS].ffill(limit=7)
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out = out.drop(columns=["water_inj", "flow_kind", "well_type"], errors="ignore")
    log.info("Volve clean: %d -> %d baris, %d sumur", n0, len(out), out["well"].nunique())
    return out


def run(raw_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "volve_daily_clean.parquet", index=False)
    return df
