"""Generator dataset portofolio pengeboran (AUGMENTED / SINTETIS).

MENGAPA ADA FILE INI
--------------------
Dataset Volve hanya memuat data produksi 7 sumur. Untuk melatih model
risiko kegagalan, biaya, durasi & SLA dibutuhkan ribuan sumur beserta AFE
(Authorization For Expenditure) dan realisasinya — data yang di dunia nyata
bersifat rahasia perusahaan. File ini membangkitkan portofolio sumur yang:

* dikalibrasi ke Volve (distribusi produksi tahun-1 sumur sukses,
  median ~255 ribu Sm3, rentang 41 rb – 1,37 jt Sm3),
* memakai hubungan fisik/operasional yang lazim di North Sea
  (NPT naik pada HPHT, rig tua, musim dingin; biaya = spread rate x hari
  + tangible; peluang sukses geologi naik dengan kualitas seismik & sumur offset),
* sengaja memberi bias optimis pada estimasi AFE (masalah bisnis yang
  ingin diselesaikan: realisasi sering meleset dari rencana).

GANTI DENGAN DATA NYATA: cukup sediakan CSV dengan kolom di PREDRILL_FEATURES
+ TARGETS, lalu set `paths.portfolio_csv` di config.yaml.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AREAS = ["North Sea South", "North Sea Central", "North Sea North", "Norwegian Sea", "Barents Sea"]
AREA_P = [0.20, 0.30, 0.25, 0.15, 0.10]
AREA_WD = {  # (median, sigma log) water depth meter
    "North Sea South": (80, 0.25), "North Sea Central": (100, 0.25), "North Sea North": (180, 0.4),
    "Norwegian Sea": (450, 0.6), "Barents Sea": (350, 0.3),
}
AREA_PRIOR = {  # tingkat sukses historis area (prior eksplorasi)
    "North Sea South": 0.45, "North Sea Central": 0.55, "North Sea North": 0.50,
    "Norwegian Sea": 0.40, "Barents Sea": 0.30,
}
AREA_FORMATIONS = {
    "North Sea South": ["Chalk", "Rotliegend"],
    "North Sea Central": ["Hugin", "Sleipner", "Heimdal", "Chalk"],
    "North Sea North": ["Brent", "Statfjord", "Heimdal"],
    "Norwegian Sea": ["Garn", "Ile", "Tilje"],
    "Barents Sea": ["Sto", "Kobbe"],
}
FORMATION_PHI = {  # (porositas rata-rata, kekerasan batuan 0-1)
    "Chalk": (0.28, 0.35), "Rotliegend": (0.14, 0.75), "Hugin": (0.21, 0.55), "Sleipner": (0.18, 0.60),
    "Heimdal": (0.26, 0.40), "Brent": (0.22, 0.55), "Statfjord": (0.19, 0.60), "Garn": (0.20, 0.55),
    "Ile": (0.18, 0.60), "Tilje": (0.17, 0.65), "Sto": (0.16, 0.65), "Kobbe": (0.15, 0.70),
}
RIG_DAYRATE = {"platform": 110_000, "jackup": 160_000, "semisub": 320_000, "drillship": 420_000}
# indeks harga rig per tahun (siklus pasar: puncak 2014, anjlok 2016-17, naik lagi 2022+)
RATE_INDEX = {y: v for y, v in zip(range(2006, 2026), [
    0.80, 0.90, 1.00, 0.95, 0.95, 1.00, 1.10, 1.20, 1.25, 1.00, 0.70, 0.65, 0.70,
    0.75, 0.70, 0.75, 0.90, 1.05, 1.10, 1.12])}

PREDRILL_FEATURES = [
    "spud_year", "spud_month", "area", "well_class", "well_profile", "formation", "rig_type",
    "water_depth_m", "target_tvd_m", "planned_md_m", "max_inclination_deg", "pore_pressure_sg",
    "bht_degc", "hpht", "porosity", "permeability_md", "net_pay_m", "seismic_quality",
    "offset_wells_5km", "dist_to_producer_km", "area_prior_success", "rig_age_yrs",
    "contractor_tier", "crew_experience_yrs", "mud_type", "n_casing_strings", "winter_spud",
    "afe_days", "afe_cost_musd", "sla_days",
]
TARGETS = ["failure", "failure_mode", "actual_days", "actual_cost_musd", "sla_breach", "first_year_oil_sm3"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(index=range(n))
    d["well_id"] = [f"W-{i:05d}" for i in range(n)]
    d["spud_year"] = rng.integers(2006, 2026, n)
    d["spud_month"] = rng.integers(1, 13, n)
    d["area"] = rng.choice(AREAS, n, p=AREA_P)
    d["well_class"] = rng.choice(["exploration", "appraisal", "development"], n, p=[0.30, 0.20, 0.50])

    med = d["area"].map(lambda a: AREA_WD[a][0]).to_numpy()
    sig = d["area"].map(lambda a: AREA_WD[a][1]).to_numpy()
    d["water_depth_m"] = np.round(med * np.exp(rng.normal(0, sig)), 0)

    d["target_tvd_m"] = np.clip(rng.normal(3300, 750, n), 1600, 5800).round(0)
    is_dev = (d["well_class"] == "development").to_numpy()
    prof_p = np.where(is_dev[:, None], [0.10, 0.45, 0.45], [0.55, 0.40, 0.05])
    u = rng.random(n)
    d["well_profile"] = np.where(u < prof_p[:, 0], "vertical",
                                 np.where(u < prof_p[:, 0] + prof_p[:, 1], "deviated", "horizontal"))
    inc = {"vertical": (0, 8), "deviated": (20, 60), "horizontal": (84, 92)}
    d["max_inclination_deg"] = [round(rng.uniform(*inc[p]), 1) for p in d["well_profile"]]
    stretch = d["well_profile"].map({"vertical": 1.02, "deviated": 1.30, "horizontal": 1.55}).to_numpy()
    lateral = np.where(d["well_profile"] == "horizontal", rng.uniform(500, 2000, n), 0)
    d["planned_md_m"] = (d["target_tvd_m"] * stretch * rng.normal(1, 0.04, n) + lateral).round(0)

    d["pore_pressure_sg"] = np.clip(1.03 + 0.00012 * (d["target_tvd_m"] - 1500) + rng.gamma(2, 0.06, n), 1.03, 2.05).round(2)
    d["bht_degc"] = (4 + 0.034 * d["target_tvd_m"] + rng.normal(0, 8, n)).round(0)
    d["hpht"] = ((d["bht_degc"] > 150) | (d["pore_pressure_sg"] > 1.70)).astype(int)

    d["formation"] = [rng.choice(AREA_FORMATIONS[a]) for a in d["area"]]
    phi_mu = d["formation"].map(lambda f: FORMATION_PHI[f][0]).to_numpy()
    hard = d["formation"].map(lambda f: FORMATION_PHI[f][1]).to_numpy()
    d["porosity"] = np.clip(phi_mu - 0.00002 * (d["target_tvd_m"] - 3000) + rng.normal(0, 0.035, n), 0.04, 0.35).round(3)
    d["permeability_md"] = np.round(np.maximum(10 ** (-1.2 + 14 * d["porosity"] + rng.normal(0, 0.45, n)), 0.1), 1)
    d["net_pay_m"] = np.round(np.exp(rng.normal(np.log(22), 0.6, n)), 1)

    d["seismic_quality"] = np.clip(np.round(rng.normal(3.2, 1.0, n)), 1, 5)
    d["offset_wells_5km"] = rng.poisson(np.where(is_dev, 7, np.where(d["well_class"] == "appraisal", 3, 0.8)))
    d["dist_to_producer_km"] = np.round(np.where(is_dev, rng.exponential(2, n), rng.exponential(18, n)) + 0.2, 1)
    d["area_prior_success"] = (d["area"].map(AREA_PRIOR) + rng.normal(0, 0.04, n)).clip(0.1, 0.9).round(3)

    wd = d["water_depth_m"].to_numpy()
    rig = np.where(wd > 1000, "drillship", np.where(wd > 130, "semisub", "jackup"))
    rig = np.where(is_dev & (rng.random(n) < 0.35) & (wd < 400), "platform", rig)
    d["rig_type"] = rig
    d["rig_age_yrs"] = rng.integers(1, 42, n)
    d["contractor_tier"] = rng.choice([1, 2, 3], n, p=[0.45, 0.40, 0.15])
    d["crew_experience_yrs"] = np.round(rng.gamma(3, 3.2, n), 1)
    d["mud_type"] = np.where((d["hpht"] == 1) | (d["well_profile"] == "horizontal") | (rng.random(n) < 0.4), "OBM", "WBM")
    d["n_casing_strings"] = np.clip(3 + (d["target_tvd_m"] > 3000) + (d["pore_pressure_sg"] > 1.5) + d["hpht"]
                                    + (rng.random(n) < 0.25), 3, 7).astype(int)
    north = d["area"].isin(["North Sea North", "Norwegian Sea", "Barents Sea"]).to_numpy()
    d["winter_spud"] = d["spud_month"].isin([11, 12, 1, 2, 3]).astype(int)

    # ---------------- DURASI ----------------
    rop_m_day = 260 * (1.25 - 0.6 * hard) * np.where(d["rig_type"] == "platform", 1.1, 1.0)
    learning = 1 - 0.012 * (d["spud_year"] - 2006)
    completion = np.where(is_dev, rng.uniform(10, 22, n), 0)
    flat_days = d["planned_md_m"] / rop_m_day + 1.6 * d["n_casing_strings"] + 4 + completion
    flat_days = flat_days * learning

    npt = (0.10 + 0.14 * d["hpht"] + 0.09 * d["winter_spud"] * north
           + 0.0035 * (d["rig_age_yrs"] - 15) + 0.06 * (d["contractor_tier"] == 3)
           - 0.010 * (d["crew_experience_yrs"] - 9).clip(-9, 12)
           + 0.07 * (d["well_profile"] == "horizontal") + 0.12 * (d["pore_pressure_sg"] > 1.55)
           + 0.04 * (d["mud_type"] == "WBM") * (d["well_profile"] != "vertical"))
    npt = np.clip(npt + rng.normal(0, 0.06, n), 0.0, 1.2)
    p_sidetrack = _sigmoid(-3.4 + 1.1 * d["hpht"] + 0.8 * (d["well_profile"] == "horizontal")
                           + 0.6 * (d["contractor_tier"] == 3) + 0.02 * (d["rig_age_yrs"] - 15))
    sidetrack = rng.random(n) < p_sidetrack
    extra = np.where(sidetrack, rng.uniform(8, 35, n), 0)
    d["actual_days"] = np.round(flat_days * (1 + npt) * np.exp(rng.normal(0, 0.08, n)) + extra, 1)

    # AFE: estimasi engineer — pakai durasi "flat" + kontingensi 10%,
    # tidak memperhitungkan penuh HPHT/cuaca/rig tua (bias optimis).
    d["afe_days"] = np.round(flat_days * (1.10 + 0.05 * d["hpht"]) * np.exp(rng.normal(0, 0.10, n)), 1)
    d["sla_days"] = np.round(d["afe_days"] * 1.10, 1)  # SLA kontrak = AFE + 10%
    d["sla_breach"] = (d["actual_days"] > d["sla_days"]).astype(int)

    # ---------------- BIAYA ----------------
    idx = d["spud_year"].map(RATE_INDEX).to_numpy()
    dayrate = d["rig_type"].map(RIG_DAYRATE).to_numpy() * idx
    spread = dayrate * 2.1  # rig + service + logistik
    tangible = 900 * d["planned_md_m"] * (1 + 0.35 * d["hpht"]) + 250_000 * d["n_casing_strings"]
    compl_cost = np.where(is_dev, rng.uniform(8e6, 18e6, n), 0)
    d["afe_cost_musd"] = np.round((d["afe_days"] * spread + tangible + compl_cost) / 1e6, 2)

    # ---------------- KEGAGALAN & PRODUKSI ----------------
    logit_geo = (np.select([d["well_class"] == "exploration", d["well_class"] == "appraisal"], [-0.35, 1.0], 2.1)
                 + 0.45 * (d["seismic_quality"] - 3) + 3.0 * (d["area_prior_success"] - 0.45)
                 + 0.12 * np.minimum(d["offset_wells_5km"], 10) - 0.25 * np.log1p(d["dist_to_producer_km"])
                 + 5.0 * (d["porosity"] - 0.19) + rng.normal(0, 0.5, n))
    geo_success = rng.random(n) < _sigmoid(logit_geo)
    p_tech = _sigmoid(-4.0 + 1.2 * d["hpht"] + 0.9 * sidetrack + 0.5 * (d["pore_pressure_sg"] > 1.6)
                      + 0.4 * (d["contractor_tier"] == 3))
    tech_fail = rng.random(n) < p_tech

    kh = d["permeability_md"] * d["net_pay_m"]
    kh_med = np.median(kh)
    log_q = (np.log(255_000) + 0.42 * np.log(kh / kh_med) + 0.55 * (d["well_profile"] == "horizontal")
             + 0.8 * (d["pore_pressure_sg"] - 1.3) - 0.35 * (d["well_class"] != "development")
             + rng.normal(0, 0.55, n))
    q = np.exp(log_q)
    # kalibrasi ke Volve: median sumur sukses development ~255 rb Sm3
    ok_dev = geo_success & ~tech_fail & is_dev
    q *= 255_000 / np.median(q[ok_dev])
    q = np.clip(q, 5_000, 2_500_000)

    subcommercial = geo_success & ~tech_fail & (q < 60_000)
    d["failure_mode"] = np.select([tech_fail, ~geo_success, subcommercial],
                                  ["technical", "dry_hole", "sub_commercial"], "success")
    d["failure"] = (d["failure_mode"] != "success").astype(int)
    d["first_year_oil_sm3"] = np.where(geo_success & ~tech_fail, q.round(0), 0.0)

    # biaya aktual: sumur gagal = plug & abandon (tanpa completion) + biaya NPT
    npt_services = rng.uniform(0.05, 0.25, n) * extra * spread
    compl_actual = np.where(is_dev & (d["failure_mode"] != "dry_hole") & ~tech_fail, compl_cost * rng.normal(1.08, 0.08, n), 0)
    pa_cost = np.where(d["failure"] == 1, rng.uniform(1.5e6, 4e6, n), 0)
    d["actual_cost_musd"] = np.round((d["actual_days"] * spread * np.exp(rng.normal(0, 0.06, n))
                                      + tangible * rng.normal(1.05, 0.07, n) + compl_actual + npt_services + pa_cost) / 1e6, 2)

    # ---------------- DATA KOTOR (realistis) ----------------
    for col, frac in [("seismic_quality", 0.08), ("crew_experience_yrs", 0.05), ("permeability_md", 0.12),
                      ("rig_age_yrs", 0.03), ("bht_degc", 0.04), ("net_pay_m", 0.03)]:
        d.loc[rng.random(n) < frac, col] = np.nan
    bad = rng.random(n) < 0.004  # kesalahan unit (ft dicatat sebagai m)
    d.loc[bad, "net_pay_m"] = d.loc[bad, "net_pay_m"] * 3.28
    neg = rng.random(n) < 0.003
    d.loc[neg, "water_depth_m"] = -d.loc[neg, "water_depth_m"]  # salah tanda

    d["data_source"] = "synthetic_calibrated_v1"
    return d


def clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi & perbaikan data master sumur (berlaku juga untuk data nyata)."""
    df = df.copy()
    df["water_depth_m"] = df["water_depth_m"].abs()
    df.loc[df["net_pay_m"] > 150, "net_pay_m"] = df.loc[df["net_pay_m"] > 150, "net_pay_m"] / 3.28
    df.loc[(df["porosity"] <= 0) | (df["porosity"] > 0.45), "porosity"] = np.nan
    for c in ["afe_days", "afe_cost_musd", "actual_days", "actual_cost_musd"]:
        if c in df:
            df.loc[df[c] <= 0, c] = np.nan
    if "sla_days" in df and df["sla_days"].isna().any():
        df["sla_days"] = df["sla_days"].fillna(df["afe_days"] * 1.10)
    return df.drop_duplicates("well_id")
