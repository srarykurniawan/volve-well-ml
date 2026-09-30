"""Lapisan ekonomi: NPV sumur sukses, Expected Monetary Value (EMV), penalti SLA."""
from __future__ import annotations

import numpy as np


def npv_success(first_year_oil_sm3, well_cost_musd, eco: dict):
    """NPV (juta USD) sumur yang berhasil produksi, decline eksponensial tahunan."""
    q1 = np.asarray(first_year_oil_sm3, float) * eco["sm3_to_bbl"]
    years = np.arange(eco["production_years"])
    decline = (1 - eco["annual_decline"]) ** years
    disc = (1 + eco["discount_rate"]) ** -(years + 0.5)
    margin = (eco["oil_price_usd_bbl"] - eco["opex_usd_bbl"]) * (1 - eco["royalty_tax_rate"])
    rev = np.outer(q1, decline * disc).sum(axis=1) * margin / 1e6
    return rev - np.asarray(well_cost_musd, float)


def emv(p_fail, first_year_oil_sm3, cost_success_musd, cost_fail_musd, eco: dict,
        p_sla=None, overrun_days=None):
    """EMV = (1-p)·NPV_sukses − p·biaya_gagal − E[penalti SLA]."""
    p = np.asarray(p_fail, float)
    v = (1 - p) * npv_success(first_year_oil_sm3, cost_success_musd, eco) - p * np.asarray(cost_fail_musd, float)
    if p_sla is not None and overrun_days is not None:
        v = v - np.asarray(p_sla) * np.asarray(overrun_days) * eco["sla_penalty_usd_per_day"] / 1e6
    return v
