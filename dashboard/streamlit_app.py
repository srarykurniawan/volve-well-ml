"""Dashboard operasional WellML.   streamlit run dashboard/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellml.predict import WellScorer  # noqa: E402

BLUE, ORANGE, INK, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "rgba(42,120,214,0.18)"
DATA_FILE = ROOT / "reports" / "dashboard_data.json"
MODEL_FILE = ROOT / "models" / "portfolio_models.joblib"
st.set_page_config(page_title="WellML — Drilling Decision", layout="wide")


@st.cache_data
def load():
    with open(DATA_FILE) as f:
        return json.load(f)


@st.cache_resource
def scorer():
    return WellScorer.load(MODEL_FILE)


def retrain_ui(reason: str):
    """Fallback bila artefak hilang / tidak kompatibel (mis. versi library berbeda di server)."""
    st.warning(f"Artefak model belum siap: {reason}")
    st.write("Latih ulang model di server (mode cepat, sekitar 3–6 menit). "
             "Solusi permanen: commit folder `models/` dan `reports/` hasil training lokal dengan versi library "
             "yang sama seperti di `requirements.txt`.")
    if st.button("Latih ulang model sekarang", type="primary"):
        from wellml.pipeline import run

        with st.spinner("Training pipeline berjalan..."):
            run(fast=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.stop()


if not DATA_FILE.exists() or not MODEL_FILE.exists():
    retrain_ui("file `reports/dashboard_data.json` atau `models/portfolio_models.joblib` tidak ditemukan.")
try:
    d = load()
    scorer()
except Exception as e:  # noqa: BLE001
    retrain_ui(f"gagal memuat artefak ({type(e).__name__}: {e}).")
m, biz = d["metrics"], d["metrics"]["business"]
st.title("WellML — Forecast & Keputusan Pengeboran")
st.caption("Produksi: data Volve (Equinor) asli · Risiko/biaya/durasi: portofolio augmented terkalibrasi — ganti dengan data AFE/DDR Anda.")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Forecast produksi", "Portofolio & SLA", "Scoring prospek baru"])

with tab1:
    b0, b1 = biz["baseline_drill_all"], biz["model_emv_gt_0"]
    c = st.columns(4)
    c[0].metric("Failure rate sumur dibor", f"{b1['failure_rate']:.0%}", f"{(b1['failure_rate'] - b0['failure_rate']) * 100:+.0f} pp vs bor semua", delta_color="inverse")
    c[1].metric("CAPEX dibutuhkan", f"${b1['capex_musd'] / 1000:.1f} B", f"{b1['capex_musd'] / b0['capex_musd'] - 1:+.0%}", delta_color="inverse")
    c[2].metric("Error estimasi biaya (MAPE)", f"{m['portfolio']['cost']['test']['LightGBM']['MAPE']:.1%}",
                f"AFE {m['portfolio']['cost']['test']['AFE']['MAPE']:.1%}", delta_color="off")
    c[3].metric("Forecast produksi 30 hari (WAPE)", f"{m['production']['test']['LightGBM']['WAPE']:.1%}",
                f"Persistence {m['production']['test']['Persistence']['WAPE']:.1%}", delta_color="off")
    st.subheader("Perbandingan model vs baseline (holdout out-of-time)")
    rows = []
    for k, lab in [("cost", "Biaya"), ("duration", "Durasi")]:
        t = m["portfolio"][k]["test"]
        rows.append({"Target": lab, "ML MAPE": t["LightGBM"]["MAPE"], "AFE MAPE": t["AFE"]["MAPE"],
                     "ML bias": t["LightGBM"]["bias_pct"], "AFE bias": t["AFE"]["bias_pct"]})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1%}" for c in ["ML MAPE", "AFE MAPE", "ML bias", "AFE bias"]}))

with tab2:
    te = pd.DataFrame(d["production_test"])
    hist = pd.DataFrame(d["production_history"])
    well = st.selectbox("Sumur", sorted(te["well"].unique()))
    g, h = te[te["well"] == well], hist[hist["well"] == well]
    fig = go.Figure()
    fig.add_scatter(x=h["date"], y=h["oil"], name="Historis (rata-rata mingguan)", line=dict(color="#898781", width=1))
    fig.add_scatter(x=g["date"], y=g["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=g["date"], y=g["p10"], fill="tonexty", fillcolor=BAND, line=dict(width=0), name="P10–P90")
    fig.add_scatter(x=g["date"], y=g["target"], name="Aktual 30 hari ke depan", line=dict(color=INK, width=1.6))
    fig.add_scatter(x=g["date"], y=g["pred_lgbm"], name="Forecast P50", line=dict(color=BLUE, width=2))
    fig.update_layout(height=460, yaxis_title="Sm³/hari", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(m["production"]["test"]).T[["WAPE", "MAE", "R2", "bias_pct"]])

with tab3:
    pt = pd.DataFrame(d["portfolio_test"])
    f = st.multiselect("Kelas sumur", sorted(pt["well_class"].unique()), default=sorted(pt["well_class"].unique()))
    pt = pt[pt["well_class"].isin(f)]
    fig = go.Figure(go.Scatter(x=pt["p_fail"], y=pt["emv_musd"], mode="markers",
                               marker=dict(size=7, color=[BLUE if x == "DRILL" else ORANGE for x in pt["decision"]]),
                               text=pt["well_id"], hovertemplate="%{text}<br>P(gagal)=%{x:.0%}<br>EMV=$%{y:.0f}M"))
    fig.update_layout(height=440, xaxis_title="Probabilitas gagal", yaxis_title="EMV (juta USD)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pt.sort_values("emv_musd", ascending=False)[
        ["well_id", "area", "well_class", "p_fail", "p_sla_breach", "cost_p50", "days_p50", "sla_days", "emv_musd", "decision"]])

with tab4:
    st.write("Masukkan parameter pra-bor untuk mendapatkan rekomendasi go/no-go.")
    c1, c2, c3 = st.columns(3)
    p = dict(
        well_id="NEW-1", spud_year=2026, spud_month=c1.slider("Bulan spud", 1, 12, 6),
        area=c1.selectbox("Area", ["North Sea South", "North Sea Central", "North Sea North", "Norwegian Sea", "Barents Sea"], 1),
        well_class=c1.selectbox("Kelas", ["exploration", "appraisal", "development"], 2),
        well_profile=c1.selectbox("Profil", ["vertical", "deviated", "horizontal"], 2),
        formation=c1.text_input("Formasi", "Hugin"),
        rig_type=c2.selectbox("Rig", ["platform", "jackup", "semisub", "drillship"], 1),
        water_depth_m=c2.number_input("Kedalaman laut (m)", 10.0, 3000.0, 95.0),
        target_tvd_m=c2.number_input("Target TVD (m)", 1000.0, 7000.0, 3200.0),
        planned_md_m=c2.number_input("Planned MD (m)", 1000.0, 10000.0, 5400.0),
        max_inclination_deg=c2.number_input("Inklinasi maks (°)", 0.0, 95.0, 88.0),
        pore_pressure_sg=c2.number_input("Pore pressure (SG)", 1.0, 2.3, 1.35),
        bht_degc=c2.number_input("BHT (°C)", 30.0, 250.0, 115.0),
        porosity=c3.number_input("Porositas", 0.03, 0.40, 0.21), permeability_md=c3.number_input("Permeabilitas (mD)", 0.1, 5000.0, 250.0),
        net_pay_m=c3.number_input("Net pay (m)", 1.0, 150.0, 22.0), seismic_quality=c3.slider("Kualitas seismik", 1, 5, 4),
        offset_wells_5km=c3.number_input("Sumur offset < 5 km", 0, 50, 6),
        dist_to_producer_km=c3.number_input("Jarak ke produsen (km)", 0.0, 200.0, 2.0),
        area_prior_success=c3.slider("Prior sukses area", 0.1, 0.9, 0.55),
        rig_age_yrs=c1.number_input("Umur rig (thn)", 0, 50, 15), contractor_tier=c1.selectbox("Tier kontraktor", [1, 2, 3], 0),
        crew_experience_yrs=c1.number_input("Pengalaman kru (thn)", 0.0, 40.0, 9.0), mud_type=c2.selectbox("Mud", ["OBM", "WBM"]),
        n_casing_strings=c2.number_input("Jumlah casing", 2, 9, 4),
        afe_days=c3.number_input("AFE hari", 5.0, 400.0, 48.0), afe_cost_musd=c3.number_input("AFE biaya (juta USD)", 1.0, 500.0, 34.0),
    )
    p["hpht"] = int(p["bht_degc"] > 150 or p["pore_pressure_sg"] > 1.7)
    p["winter_spud"] = int(p["spud_month"] in (11, 12, 1, 2, 3))
    if st.button("Hitung", type="primary"):
        r = scorer().score(pd.DataFrame([p])).iloc[0]
        k = st.columns(5)
        k[0].metric("Keputusan", r["decision"])
        k[1].metric("P(gagal)", f"{r['p_fail']:.0%}")
        k[2].metric("P(langgar SLA)", f"{r['p_sla_breach']:.0%}")
        k[3].metric("Biaya P50 (P10–P90)", f"${r['cost_p50']:.1f}M", f"{r['cost_p10']:.1f}–{r['cost_p90']:.1f}", delta_color="off")
        k[4].metric("EMV", f"${r['emv_musd']:.0f}M")
        st.json(r.to_dict())
