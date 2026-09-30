"""(Opsional) Unduh data wellbore eksplorasi ASLI dari Sodir FactPages (ex-NPD).

Jalankan di komputer Anda (butuh akses internet ke factpages.sodir.no):

    python -m wellml.data.download_sodir --out data/raw/sodir_exploration.csv

Data ini (±2.000 sumur eksplorasi Norwegia) berisi hasil (wlbContent: OIL/GAS/DRY/SHOWS),
tanggal spud & selesai, kedalaman, kedalaman laut, formasi, dll. Fungsi `to_portfolio_schema`
memetakan kolomnya ke skema PREDRILL_FEATURES sehingga model risiko dry-hole dan durasi
dapat dilatih ulang pada data nyata. Kolom biaya/AFE tidak tersedia publik -> isi dari
data internal perusahaan.
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd

URL = ("https://factpages.sodir.no/public?/Factpages/external/tableview/wellbore_exploration_all"
       "&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en"
       "&rs:Format=CSV&Top100=false")


def download() -> pd.DataFrame:
    import urllib.request

    with urllib.request.urlopen(URL, timeout=120) as r:  # noqa: S310 (URL tetap)
        raw = r.read().decode("utf-8-sig")
    return pd.read_csv(io.StringIO(raw))


def to_portfolio_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["well_id"] = df["wlbWellboreName"]
    entry = pd.to_datetime(df["wlbEntryDate"], dayfirst=True, errors="coerce")
    compl = pd.to_datetime(df["wlbCompletionDate"], dayfirst=True, errors="coerce")
    out["spud_year"] = entry.dt.year
    out["spud_month"] = entry.dt.month
    out["winter_spud"] = entry.dt.month.isin([11, 12, 1, 2, 3]).astype(int)
    out["area"] = df.get("wlbMainArea")
    out["well_class"] = df.get("wlbPurpose", pd.Series(dtype=str)).str.lower().map(
        {"wildcat": "exploration", "appraisal": "appraisal"}).fillna("exploration")
    out["water_depth_m"] = pd.to_numeric(df.get("wlbWaterDepth"), errors="coerce")
    out["target_tvd_m"] = pd.to_numeric(df.get("wlbFinalVerticalDepth"), errors="coerce")
    out["planned_md_m"] = pd.to_numeric(df.get("wlbTotalDepth"), errors="coerce")
    out["max_inclination_deg"] = pd.to_numeric(df.get("wlbMaxInclation"), errors="coerce")
    out["bht_degc"] = pd.to_numeric(df.get("wlbBottomHoleTemperature"), errors="coerce")
    out["formation"] = df.get("wlbFormationAtTd")
    out["actual_days"] = (compl - entry).dt.days.astype(float)
    content = df.get("wlbContent", pd.Series(dtype=str)).fillna("").str.upper()
    out["failure"] = (~content.str.contains("OIL|GAS")).astype(int)
    out["failure_mode"] = np.where(out["failure"] == 1, "dry_hole", "success")
    out["data_source"] = "sodir_factpages"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/sodir_exploration.csv")
    a = ap.parse_args()
    raw = download()
    raw.to_csv(a.out, index=False)
    to_portfolio_schema(raw).to_csv(a.out.replace(".csv", "_mapped.csv"), index=False)
    print(f"Tersimpan: {a.out} ({len(raw)} baris)")
