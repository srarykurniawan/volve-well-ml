# WellML — End-to-End Forecasting & Decision Support untuk Proyek Pengeboran Sumur Minyak

> Studi kasus: **Volve Field (Equinor, Laut Utara)** + portofolio pengeboran augmented.
> Tujuan bisnis: mengurangi sumur gagal, memperbaiki akurasi prediksi produksi & biaya, dan mengendalikan risiko keterlambatan (SLA).

---

## 1. Problem statement

| Masalah di lapangan | Pertanyaan ML | Model |
|---|---|---|
| Produksi aktual meleset dari prediksi | Berapa produksi minyak 30 hari ke depan per sumur (+ ketidakpastian)? | **M1** Forecast produksi (LightGBM + CQR, data Volve asli) |
| Banyak sumur gagal (dry hole / teknis / sub-komersial) | Berapa peluang sumur ini gagal *sebelum* dibor? | **M2** Klasifikasi risiko kegagalan (terkalibrasi) |
| Biaya operasional membengkak dari AFE | Berapa biaya realistis (P10/P50/P90)? | **M3** Regresi biaya |
| Durasi molor, melanggar SLA | Berapa hari pengeboran & peluang melanggar SLA? | **M4** Regresi durasi + **M5** klasifikasi SLA breach |
| Prediksi produksi pra-bor tidak akurat | Berapa produksi tahun-1 jika sukses? | **M6** Regresi produksi pra-bor |
| Keputusan go/no-go masih intuitif | Prospek mana yang layak dibor? | **EMV** = (1−p)·NPV − p·biaya_gagal − E[penalti SLA] |

## 2. Data

| Dataset | Sumber | Isi | Dipakai untuk |
|---|---|---|---|
| `data/raw/volve_production.xlsx` | Equinor Volve Data Village (open data) | 15.634 baris produksi harian, 7 wellbore, 2007–2016 (tekanan & suhu downhole, choke, WHP, oil/gas/water) | M1 |
| `data/processed/drilling_portfolio.csv` | **Augmented/sintetis**, dibangkitkan `wellml.data.drilling_portfolio` | 2.500 sumur, 30 fitur pra-bor, AFE, SLA, realisasi durasi/biaya/kegagalan/produksi | M2–M6 |
| (opsional) Sodir FactPages | `python -m wellml.data.download_sodir` | ±2.000 sumur eksplorasi Norwegia asli (hasil OIL/GAS/DRY, tanggal, kedalaman) | Validasi M2/M4 dengan data nyata |

**Penting — kejujuran data.** Volve hanya memuat produksi 7 sumur; data biaya, AFE, dan SLA di dunia nyata bersifat rahasia. Portofolio augmented dikalibrasi ke Volve (median produksi tahun-1 sumur sukses ≈ 255 ribu Sm³, sama dengan median sumur Volve) dan memakai hubungan fisik/operasional North Sea (NPT naik pada HPHT, rig tua, musim dingin; biaya = spread-rate × hari + tangible). Angka metrik M2–M6 **mendemonstrasikan metodologi**, bukan klaim performa di lapangan Anda. Untuk produksi: ganti CSV dengan data DDR/AFE perusahaan Anda (skema = `PREDRILL_FEATURES` + `TARGETS` di `drilling_portfolio.py`), lalu set `paths.portfolio_csv` di `config/config.yaml`. Pipeline otomatis memakai file tersebut.

## 3. Arsitektur

```
data/raw ──► ingest & cleaning ──► feature engineering ──► Optuna tuning ──► training ──► evaluasi out-of-time
                (audit rules)        (anti-leakage)        (CV walk-forward/   (champion      (vs baseline bisnis)
                                                            stratified)         selection)         │
                                                                                                   ▼
             models/*.joblib ◄── kalibrasi (isotonic, CQR) ◄── explainability (SHAP) ◄── reports/metrics.json
                   │
                   ├─► wellml.predict.WellScorer  (Python)
                   ├─► api/app.py                 (FastAPI  POST /score)
                   └─► dashboard/streamlit_app.py (Streamlit)
```

```
volve-well-ml/
├── config/config.yaml              # semua parameter (horizon, split, ekonomi, SLA penalty)
├── data/raw/                       # Volve asli
├── src/wellml/
│   ├── data/ingest_volve.py        # cleaning Volve (shut-in, outlier IQR, kalender harian)
│   ├── data/drilling_portfolio.py  # generator + validasi data master sumur
│   ├── data/download_sodir.py      # pengunduh data wellbore asli (opsional)
│   ├── features/production_features.py
│   ├── models/production_forecast.py   # M1: Persistence, Arps DCA, LightGBM + CQR
│   ├── models/portfolio_models.py      # M2–M6 + evaluasi bisnis
│   ├── economics.py                # NPV, EMV, penalti SLA
│   ├── evaluation/metrics.py
│   ├── reporting.py                # figur PNG
│   ├── pipeline.py                 # orkestrasi end-to-end
│   └── predict.py                  # WellScorer (inference)
├── api/app.py                      # FastAPI dengan validasi input (pydantic)
├── dashboard/streamlit_app.py
├── tests/test_wellml.py            # 9 test: leakage, reproducibility, transform, API
├── models/  reports/               # artefak hasil training
├── Dockerfile  Makefile  pyproject.toml
```

## 4. Metodologi (poin robustness)

**Cleaning Volve** — hanya aliran produksi; F-4/F-5 (injektor) dibuang; sensor bernilai 0 saat shut-in → missing; clip outlier IQR×5 per sumur; kalender harian lengkap; forward-fill sensor maks. 7 hari.

**M1 Forecast produksi**
- Target: rata-rata volume minyak harian t+1…t+30. Semua fitur ≤ t (diuji di `test_production_target_is_future_only`).
- Model memprediksi **perubahan relatif** `log((y+1)/(mean30+1))`, bukan level absolut — tree model tidak bisa mengekstrapolasi level yang terus turun (versi level absolut kalah dari persistence: WAPE 0,37 vs 0,20).
- Walk-forward CV dengan **purge gap** = horizon (mencegah overlap target train/validasi), Optuna 25 trial.
- Interval P10–P90: LightGBM quantile + **Conformalized Quantile Regression** (kalibrasi pada 365 hari terakhir train).
- Baseline: persistence & Arps hyperbolic DCA (fit pada 540 hari terakhir).

**M2–M6 Portofolio**
- Hanya fitur yang tersedia **sebelum spud** (assert anti-leakage di `engineer()`).
- Split **out-of-time**: latih spud ≤ 2021 (2.007 sumur), uji ≥ 2022 (493 sumur).
- Optuna (25 trial) + 5-fold CV; LightGBM menangani missing & kategori secara native.
- Klasifikasi: **champion selection** di antara LightGBM terkalibrasi isotonic, Logistic Regression, dan blend — dipilih berdasar log-loss OOF, bukan di test set.
- Threshold M2 dipilih untuk **memaksimalkan nilai ekonomi** portofolio pada OOF, bukan F1.
- Regresi pada log-target + quantile P10/P50/P90 + CQR out-of-time.
- SHAP untuk menjelaskan driver setiap model.

## 5. Hasil (holdout out-of-time)

### M1 — Forecast produksi 30 hari (Volve asli, Okt-2015 → Agu-2016, 1.466 titik forecast)

| Model | WAPE | MAE (Sm³/hari) | R² |
|---|---|---|---|
| **LightGBM (relatif)** | **17,3%** | **48,6** | **0,94** |
| Persistence | 20,1% | 56,4 | 0,91 |
| Arps DCA | 62,1% | 174,2 | 0,22 |

Cakupan interval P10–P90: **85%** (nominal 80%). Driver utama: level lag terakhir relatif, tren 30/90 hari, uptime, water cut, suhu downhole.

### M2–M6 — Portofolio (data augmented, 493 sumur uji)

| Model | Metrik | ML | Baseline |
|---|---|---|---|
| M2 Risiko gagal (champion: blend LGBM+LR) | ROC-AUC / Brier | **0,837 / 0,163** | LR 0,830 |
| M5 SLA breach (champion: Logistic Regression) | ROC-AUC / recall | **0,782 / 85%** | LGBM 0,753 |
| M3 Biaya | MAPE / bias | **12,4% / −0,2%** | AFE 15,3% / −11,6% |
| M4 Durasi | MAPE / bias | **12,7% / +2,1%** | AFE 14,9% / −12,0% |
| M6 Produksi tahun-1 | WAPE / R² | **43% / 0,64** | Analog 69% / 0,01 |

Bias AFE sekitar −12% (estimasi terlalu optimis) adalah akar masalah "biaya & durasi selalu lebih dari rencana"; model menghilangkan bias tersebut.

### Dampak keputusan (simulasi pada test set)

| Strategi | Sumur dibor | Failure rate | CAPEX |
|---|---|---|---|
| Bor semua prospek | 493 | 43,8% | $15,0 B |
| **Bor jika EMV > 0** | **317** | **26,5%** | **$9,7 B (−35%)** |

Nilai terealisasi portofolio naik ±4% dengan CAPEX 35% lebih rendah.

## 6. Cara menjalankan (virtual environment)

Butuh **Python 3.11 atau 3.12**.

**Windows (PowerShell)**
```powershell
cd volve-well-ml
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # jika diblokir: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps            # daftarkan paket wellml
```

**macOS / Linux**
```bash
cd volve-well-ml
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

Lalu:
```bash
python -m wellml.pipeline          # latih ulang semua model (±5 menit; --fast ±2 menit)
pytest -q                          # 9 test
streamlit run dashboard/streamlit_app.py   # http://localhost:8501
uvicorn api.app:app --port 8000    # http://localhost:8000/docs
deactivate                         # keluar dari venv
```

## 7. Deploy ke Streamlit Community Cloud

File yang sudah disiapkan: `requirements.txt` (versi di-pin sama dengan saat training), `packages.txt` (`libgomp1` untuk LightGBM), `.streamlit/config.toml`, `.gitignore` (tidak mengabaikan `models/` & `reports/`).

1. Buat repository baru di GitHub (public atau private), lalu dari folder project:
   ```bash
   git init
   git add .
   git commit -m "WellML: forecasting & drilling decision dashboard"
   git branch -M main
   git remote add origin https://github.com/<username>/volve-well-ml.git
   git push -u origin main
   ```
   Pastikan `models/*.joblib`, `reports/dashboard_data.json`, dan `data/raw/volve_production.xlsx` ikut ter-push.
2. Buka https://share.streamlit.io → login dengan GitHub → **Create app** → **Deploy a public app from GitHub**.
3. Isi: Repository `<username>/volve-well-ml`, Branch `main`, Main file path `dashboard/streamlit_app.py`.
4. **Advanced settings → Python version: 3.11** (harus sama dengan versi saat model dilatih).
5. Klik **Deploy**. Build pertama ±3–5 menit.

Jika Anda melatih ulang model secara lokal, commit ulang `models/` dan `reports/`, lalu push; aplikasi akan redeploy otomatis. Bila model tidak bisa dimuat di server (misalnya versi library berbeda), dashboard menampilkan tombol **Latih ulang model sekarang** yang menjalankan pipeline mode cepat langsung di server.

Contoh API:

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '[{
  "area":"North Sea Central","well_class":"development","well_profile":"horizontal","formation":"Hugin",
  "rig_type":"jackup","water_depth_m":91,"target_tvd_m":3100,"planned_md_m":5200,"max_inclination_deg":89,
  "pore_pressure_sg":1.3,"porosity":0.22,"permeability_md":300,"net_pay_m":25,"seismic_quality":4,
  "offset_wells_5km":8,"dist_to_producer_km":1.5,"area_prior_success":0.55,"afe_days":45,"afe_cost_musd":32}]'
```

Output: `p_fail`, `p_sla_breach`, `cost_p10/50/90`, `days_p10/50/90`, `oil_p10/50/90`, `npv_if_success_musd`, `expected_sla_penalty_musd`, `emv_musd`, `decision`.

## 8. Keterbatasan & roadmap

1. **Ganti data augmented dengan data nyata** (DDR/WellView, AFE, kontrak rig) — prioritas utama sebelum dipakai untuk keputusan.
2. M1 dilatih pada 5 sumur satu lapangan; untuk lapangan lain perlu retraining dan fitur reservoir (PVT, tekanan statik).
3. Tambah model produksi fisika-informed (material balance / CRM) untuk horizon > 1 tahun.
4. MLOps: registry model (MLflow), monitoring drift (PSI) per kuartal, retraining terjadwal.
5. Optimasi portofolio dengan batasan anggaran & rig (integer programming) di atas EMV.

**Lisensi data**: Volve data © Equinor, dirilis di bawah Equinor Open Data Licence — cantumkan atribusi saat dipublikasikan.
