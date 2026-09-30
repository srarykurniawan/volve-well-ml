# Glosarium — WellML (Volve Drilling Intelligence)

Istilah dan singkatan yang dipakai di project, dashboard, dan README.

## 1. Istilah migas & pengeboran

| Istilah | Kepanjangan | Pengertian singkat |
|---|---|---|
| **AFE** | Authorization for Expenditure | Anggaran biaya dan durasi yang disetujui sebelum sumur dibor. Di project ini dipakai sebagai pembanding (baseline). |
| **Spud** | – | Saat mata bor pertama kali menembus tanah/dasar laut, awal resmi pengeboran. |
| **Dry hole** | – | Sumur yang dibor tetapi tidak menemukan hidrokarbon dalam jumlah berarti. |
| **Sub-commercial** | – | Sumur menemukan minyak, tetapi volumenya terlalu kecil untuk menguntungkan. |
| **Technical failure** | – | Sumur gagal mencapai target karena masalah teknis (pipa terjepit, well control, runtuh). |
| **NPT** | Non-Productive Time | Waktu rig yang terbuang karena masalah, cuaca, atau menunggu peralatan. Penyebab utama jadwal molor. |
| **Sidetrack** | – | Membor cabang baru dari lubang yang sudah ada, biasanya karena lubang awal bermasalah. |
| **HPHT** | High Pressure High Temperature | Sumur bertekanan dan bersuhu sangat tinggi (umumnya >150 °C atau tekanan pori tinggi). Lebih berisiko dan mahal. |
| **TVD** | True Vertical Depth | Kedalaman vertikal lurus dari permukaan ke target. |
| **MD** | Measured Depth | Panjang lintasan lubang bor sebenarnya. Pada sumur miring/horizontal, MD > TVD. |
| **BHT** | Bottom Hole Temperature | Suhu di dasar sumur. |
| **Pore pressure (SG)** | Specific Gravity | Tekanan fluida di pori batuan, dinyatakan setara berat jenis lumpur. |
| **Porositas** | – | Persentase ruang kosong di batuan yang bisa menyimpan minyak/gas. |
| **Permeabilitas (mD)** | milliDarcy | Kemampuan batuan mengalirkan fluida. Makin tinggi, makin mudah minyak mengalir. |
| **Net pay** | – | Tebal lapisan reservoir yang benar-benar produktif. |
| **k·h** | permeabilitas × ketebalan | Indikator kapasitas aliran reservoir; berkorelasi kuat dengan produksi. |
| **Offset well** | – | Sumur lain di sekitar prospek yang datanya bisa dijadikan acuan. |
| **Exploration / Appraisal / Development** | – | Sumur pencarian cadangan baru / sumur untuk memastikan ukuran cadangan / sumur produksi di lapangan yang sudah terbukti. |
| **Jackup, Semisub, Drillship, Platform** | – | Jenis rig: berkaki ke dasar laut (perairan dangkal) / terapung semi-selam / kapal bor (laut dalam) / rig di anjungan tetap. |
| **OBM / WBM** | Oil-Based Mud / Water-Based Mud | Jenis lumpur pengeboran berbasis minyak atau air. |
| **Casing string** | – | Rangkaian pipa baja yang dipasang untuk memperkuat lubang sumur. |
| **Sm³** | Standard cubic meter | Satuan volume minyak/gas pada kondisi standar. 1 Sm³ minyak ≈ 6,29 barel. |
| **bbl** | barrel | Satuan volume minyak (≈159 liter). |
| **On-stream hours / Uptime** | – | Jam sumur benar-benar berproduksi dalam sehari / persentasenya. |
| **Choke** | – | Katup pengatur laju aliran di kepala sumur. |
| **WHP / WHT** | Wellhead Pressure / Temperature | Tekanan dan suhu di kepala sumur. |
| **Water cut** | – | Persentase air dalam cairan yang diproduksi. Naik seiring sumur menua. |
| **GOR** | Gas-Oil Ratio | Perbandingan volume gas terhadap minyak yang diproduksi. |
| **Decline curve / DCA** | Decline Curve Analysis | Metode klasik memperkirakan penurunan produksi sumur dari waktu ke waktu. |
| **Arps** | – | Persamaan decline curve paling umum (eksponensial, hiperbolik, harmonik). |
| **EUR** | Estimated Ultimate Recovery | Perkiraan total minyak yang akan diproduksi sumur sepanjang umurnya. |
| **DDR** | Daily Drilling Report | Laporan harian operasi pengeboran; sumber data NPT dan durasi. |
| **Volve** | – | Lapangan minyak Equinor di Laut Utara Norwegia (2008–2016); datanya dibuka untuk publik. |
| **NCS** | Norwegian Continental Shelf | Landas kontinen Norwegia, wilayah migas lepas pantai Norwegia. |
| **Sodir (ex-NPD)** | Norwegian Offshore Directorate | Regulator migas Norwegia yang mempublikasikan data sumur (FactPages). |

## 2. Istilah bisnis & ekonomi

| Istilah | Kepanjangan | Pengertian singkat |
|---|---|---|
| **SLA** | Service Level Agreement | Batas waktu yang disepakati. Di project ini = AFE hari + 10%. |
| **SLA breach** | – | Pengeboran selesai melewati batas SLA. |
| **CAPEX** | Capital Expenditure | Belanja modal, misalnya biaya mengebor sumur. |
| **OPEX** | Operational Expenditure | Biaya operasional rutin per barel produksi. |
| **NPV** | Net Present Value | Nilai keuntungan masa depan yang dikonversi ke nilai hari ini dikurangi investasi. |
| **EMV** | Expected Monetary Value | Nilai harapan yang memperhitungkan risiko: (1 − P gagal) × NPV sukses − P gagal × biaya sumur. EMV > 0 = layak dibor. |
| **Discount rate** | – | Tingkat bunga untuk menghitung nilai sekarang dari uang di masa depan (di project: 10%). |
| **Go / No-go** | – | Keputusan lanjut mengebor atau tidak. |
| **Baseline** | – | Metode pembanding yang dipakai saat ini (AFE, persistence, decline curve). Model harus mengalahkannya. |

## 3. Istilah data science & machine learning

| Istilah | Kepanjangan | Pengertian singkat |
|---|---|---|
| **End-to-end** | – | Project lengkap dari data mentah sampai aplikasi yang bisa dipakai. |
| **Pipeline** | – | Rangkaian langkah otomatis: ambil data → bersihkan → fitur → latih → evaluasi → simpan. |
| **Feature / Fitur** | – | Variabel input untuk model, misalnya kedalaman, porositas, umur rig. |
| **Feature engineering** | – | Membuat fitur baru yang lebih informatif dari data mentah (misalnya lag, rata-rata bergulir, k·h). |
| **Target** | – | Nilai yang ingin diprediksi (produksi, biaya, gagal/tidak). |
| **Lag / Rolling mean** | – | Nilai hari-hari sebelumnya / rata-rata dalam jendela waktu tertentu (7, 30, 90 hari). |
| **Data leakage** | – | Kebocoran informasi masa depan ke data latih sehingga model terlihat bagus padahal tidak. Dicegah di project ini. |
| **Train / Test split** | – | Pembagian data untuk melatih dan menguji model. |
| **Out-of-time validation** | – | Menguji model pada periode waktu setelah data latih, meniru kondisi nyata. |
| **Cross-validation (CV)** | – | Menguji model berulang kali pada potongan data berbeda agar hasilnya stabil. |
| **Walk-forward CV** | – | CV untuk data deret waktu: latih di masa lalu, uji di periode berikutnya, terus maju. |
| **Purge gap** | – | Jeda antara data latih dan uji agar target yang tumpang tindih tidak bocor. |
| **OOF** | Out-of-Fold | Prediksi untuk data yang tidak dipakai melatih model pada fold tersebut; dipakai untuk memilih model dan threshold. |
| **Hyperparameter tuning** | – | Mencari pengaturan model terbaik (jumlah pohon, kedalaman, dll.). |
| **Optuna** | – | Library Python untuk tuning hyperparameter secara otomatis dan efisien. |
| **LightGBM** | Light Gradient Boosting Machine | Algoritma berbasis banyak pohon keputusan; cepat dan akurat untuk data tabel. |
| **Logistic Regression** | – | Model klasifikasi sederhana dan mudah dijelaskan; menang untuk prediksi SLA. |
| **Blend / Ensemble** | – | Menggabungkan prediksi beberapa model (di sini rata-rata LightGBM + Logistic Regression). |
| **Champion model** | – | Model terbaik yang dipilih lewat validasi, lalu dipakai di produksi. |
| **Classification / Regression** | – | Memprediksi kategori (gagal/tidak) / memprediksi angka (biaya, hari). |
| **Quantile regression** | – | Memprediksi rentang, bukan satu angka: P10, P50, P90. |
| **P10 / P50 / P90** | – | Perkiraan rendah / tengah / tinggi. Idealnya 80% nilai aktual jatuh di antara P10 dan P90. |
| **Conformal prediction / CQR** | Conformalized Quantile Regression | Teknik mengoreksi lebar rentang P10–P90 agar cakupannya sesuai target. |
| **Calibration** | – | Memastikan probabilitas jujur: prediksi "70% gagal" memang gagal sekitar 70% kali. |
| **Isotonic calibration** | – | Metode kalibrasi probabilitas yang fleksibel (tidak mengasumsikan bentuk kurva). |
| **Threshold** | – | Batas probabilitas untuk mengambil keputusan (misalnya gagal jika P ≥ 0,72). |
| **SHAP** | SHapley Additive exPlanations | Metode menjelaskan seberapa besar tiap fitur memengaruhi prediksi. |
| **Persistence** | – | Baseline "besok sama dengan rata-rata 30 hari terakhir". |
| **Synthetic / augmented data** | – | Data buatan yang dikalibrasi ke data nyata; dipakai karena data AFE perusahaan rahasia. |
| **Outlier** | – | Nilai ekstrem yang tidak wajar; di-clip memakai aturan IQR. |
| **IQR** | Interquartile Range | Selisih kuartil ke-3 dan ke-1; dasar deteksi outlier. |
| **Forward fill** | – | Mengisi nilai kosong dengan nilai terakhir yang diketahui. |

## 4. Metrik evaluasi

| Istilah | Kepanjangan | Pengertian singkat |
|---|---|---|
| **MAE** | Mean Absolute Error | Rata-rata selisih absolut prediksi dan aktual, dalam satuan asli. |
| **RMSE** | Root Mean Squared Error | Seperti MAE, tetapi lebih menghukum kesalahan besar. |
| **MAPE** | Mean Absolute Percentage Error | Rata-rata error dalam persen. 12% = rata-rata meleset 12%. |
| **WAPE** | Weighted Absolute Percentage Error | Total error ÷ total aktual. Lebih stabil dari MAPE saat ada nilai kecil. |
| **R²** | Coefficient of determination | Seberapa besar variasi data dijelaskan model (1 = sempurna, 0 = tidak lebih baik dari rata-rata). |
| **Bias** | – | Kecenderungan prediksi terlalu tinggi (+) atau terlalu rendah (−). AFE: −12%. |
| **ROC-AUC** | Area Under the ROC Curve | Kemampuan model membedakan sumur gagal vs sukses (0,5 = acak, 1 = sempurna). |
| **PR-AUC** | Area Under Precision-Recall Curve | Seperti ROC-AUC, tetapi fokus pada kelas positif; berguna bila data tidak seimbang. |
| **Precision** | – | Dari yang diprediksi gagal, berapa persen yang benar gagal. |
| **Recall** | – | Dari yang benar gagal, berapa persen yang berhasil terdeteksi. |
| **F1** | – | Rata-rata harmonik precision dan recall. |
| **Brier score** | – | Error kuadrat probabilitas; makin kecil makin baik dan makin terkalibrasi. |
| **Log-loss** | – | Ukuran kualitas probabilitas; dipakai untuk memilih champion model. |
| **Coverage** | – | Persentase nilai aktual yang jatuh di dalam rentang P10–P90. |
| **Pinball loss** | – | Metrik untuk menilai akurasi prediksi kuantil. |
| **Base rate** | – | Proporsi kejadian di data (misalnya 44% sumur uji gagal). |

## 5. Istilah teknis & deployment

| Istilah | Kepanjangan | Pengertian singkat |
|---|---|---|
| **venv** | virtual environment | Lingkungan Python terisolasi khusus project, agar versi library tidak bentrok. |
| **requirements.txt** | – | Daftar library beserta versinya yang harus diinstal. |
| **pip** | – | Alat instal library Python. |
| **Joblib** | – | Format penyimpanan model terlatih (`.joblib`). |
| **API / REST API** | Application Programming Interface | Pintu agar aplikasi lain bisa meminta prediksi lewat internet. |
| **FastAPI** | – | Framework Python untuk membuat API; dokumentasi otomatis di `/docs`. |
| **Endpoint** | – | Alamat spesifik di API, misalnya `POST /score`. |
| **Streamlit** | – | Framework Python untuk membuat dashboard web interaktif dengan cepat. |
| **Streamlit Community Cloud** | – | Layanan gratis untuk meng-hosting aplikasi Streamlit dari GitHub. |
| **Docker** | – | Membungkus aplikasi beserta lingkungannya agar berjalan sama di mana saja. |
| **Git / GitHub** | – | Sistem kontrol versi / layanan penyimpanan repository online. |
| **Commit / Push** | – | Menyimpan snapshot perubahan / mengirimnya ke GitHub. |
| **Unit test / pytest** | – | Pengujian otomatis bagian-bagian kode. |
| **MLOps** | Machine Learning Operations | Praktik mengelola model di produksi: versi, monitoring, retraining. |
| **Drift** | – | Perubahan pola data dari waktu ke waktu yang membuat performa model turun. |
| **PSI** | Population Stability Index | Metrik untuk mendeteksi drift. |
