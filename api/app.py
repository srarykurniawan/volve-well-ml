"""REST API scoring prospek sumur.

    uvicorn api.app:app --reload --port 8000
    # dokumentasi interaktif: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wellml.predict import WellScorer

app = FastAPI(title="WellML — Drilling Decision API", version="1.0.0",
              description="Prediksi risiko gagal, biaya, durasi, risiko SLA, produksi & EMV sumur minyak.")
_scorer: WellScorer | None = None


def scorer() -> WellScorer:
    global _scorer
    if _scorer is None:
        _scorer = WellScorer.load()
    return _scorer


class Prospect(BaseModel):
    well_id: str = "PROSPECT-001"
    spud_year: int = Field(2026, ge=2000, le=2040)
    spud_month: int = Field(6, ge=1, le=12)
    area: Literal["North Sea South", "North Sea Central", "North Sea North", "Norwegian Sea", "Barents Sea"]
    well_class: Literal["exploration", "appraisal", "development"]
    well_profile: Literal["vertical", "deviated", "horizontal"]
    formation: str = "Hugin"
    rig_type: Literal["platform", "jackup", "semisub", "drillship"]
    water_depth_m: float = Field(..., gt=0)
    target_tvd_m: float = Field(..., gt=500)
    planned_md_m: float = Field(..., gt=500)
    max_inclination_deg: float = Field(..., ge=0, le=95)
    pore_pressure_sg: float = Field(..., ge=1.0, le=2.3)
    bht_degc: float | None = None
    hpht: int = Field(0, ge=0, le=1)
    porosity: float = Field(..., gt=0, lt=0.45)
    permeability_md: float | None = None
    net_pay_m: float | None = None
    seismic_quality: float | None = Field(None, ge=1, le=5)
    offset_wells_5km: int = Field(0, ge=0)
    dist_to_producer_km: float = Field(..., ge=0)
    area_prior_success: float = Field(..., ge=0, le=1)
    rig_age_yrs: float | None = None
    contractor_tier: int = Field(2, ge=1, le=3)
    crew_experience_yrs: float | None = None
    mud_type: Literal["OBM", "WBM"] = "OBM"
    n_casing_strings: int = Field(4, ge=2, le=9)
    winter_spud: int = Field(0, ge=0, le=1)
    afe_days: float = Field(..., gt=0)
    afe_cost_musd: float = Field(..., gt=0)
    sla_days: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(prospects: list[Prospect]):
    if not prospects:
        raise HTTPException(400, "Daftar prospek kosong")
    df = pd.DataFrame([p.model_dump() for p in prospects])
    return scorer().score(df).to_dict("records")
