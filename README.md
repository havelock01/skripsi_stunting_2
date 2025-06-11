# Skripsi: Klasifikasi Efektivitas Intervensi Stunting di Desa

## 📖 Judul

**Klasifikasi Efektivitas Respon Intervensi Stunting di Desa Menggunakan Algoritma Decision Tree dan Random Forest (Studi Kasus: Data Kementerian Desa PDTT 2023)**

---

## 📊 Deskripsi

Proyek ini merupakan bagian dari skripsi S1 Informatika dengan. Proyek ini merupakan aplikasi klasifikasi berbasis _machine learning_ yang bertujuan untuk memprediksi efektivitas intervensi stunting berdasarkan data layanan desa. Dibangun menggunakan Python dan Streamlit, proyek ini mengadopsi model _Decision Tree_ dan _Random Forest_ untuk mengevaluasi keberhasilan konvergensi stunting di desa berdasarkan data tahun 2023 dari Kementerian Desa PDTT.

---

## 📂 Struktur Direktori

```
skripsi_stunting/
│   README.md
│   main.py
│   requirements.txt
|   test_env.py
│
├───data/
│   ├── jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx
│   ├── stunting_2023_cleaned.csv
│   └── stunting_2023_labeled.csv
│
├───model/
│   ├── decision_tree_model.pkl
│   └── random_forest_model.pkl
│
├───notebooks/
│   └── eksplorasi_data.ipynb
│
├───scripts/
│   ├── clean_excel_to_csv.py
│   └── labeling_efektivitas.py
│
├───src/
│   ├── encoder_utils.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── form_utils.py
│   ├── model_training.py
│   ├── preprocessing.py
│   └── shap_utils.py
│
└───streamlit_app/
    └── app.py
```

---

## 📅 Tahapan Proyek

1. **Preprocessing & Labeling**

   - Konversi Excel ke CSV (`clean_excel_to_csv.py`)
   - Labeling efektivitas berdasarkan skor (`labeling_efektivitas.py`)

2. **Pelatihan Model**

   - Script di `model_training.py` untuk melatih dan menyimpan model DT & RF

3. **Visualisasi dan Evaluasi**

   - Aplikasi Streamlit `app.py` menampilkan:

     - Interpretasi SHAP
     - Form prediksi manual
     - Distribusi prediksi
     - Evaluasi akurasi dan visualisasi confusion matrix
     - Ekspor hasil ke Excel & PDF

---

## 🛠️ Instalasi

1. **Buat environment (disarankan via conda):**

```bash
conda create -n stunting-ml python=3.10
conda activate stunting-ml
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Jalankan Aplikasi Streamlit:**

```bash
streamlit run streamlit_app/app.py
```

---

## ⚙️ Kebutuhan Environment

Lihat file `requirements.txt` atau gunakan `test_env.py` untuk memverifikasi environment yang dibutuhkan:

```bash
python test_env.py
```

## 🌐 Dependencies Utama

- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- shap, joblib
- streamlit, fpdf

---

## 🧠 Model & Algoritma

- Decision Tree Classifier
- Random Forest Classifier
- Label Encoding untuk fitur kategorikal
- SHAP untuk interpretasi model
- Evaluasi: Confusion Matrix, Classification Report, F1-Score, Distribusi Probabilitas

## 🧪 Dokumentasi Teknis

### `model_training.py`

- `train_model_rf(X, y)` — latih model Random Forest dan simpan ke `.pkl`
- `train_model_dt(X, y)` — latih model Decision Tree dan simpan ke `.pkl`

### `encoder_utils.py`

- `fit_label_encoders(df, fitur_kat)` — latih `LabelEncoder` untuk setiap kolom
- `normalize_kategorikal(df, fitur_kat)` — normalisasi nilai kategorikal

### `evaluation.py`

- `evaluate_model(model, X, y)` — kembalikan `classification_report`, `confusion_matrix`

### `shap_utils.py`

- `generate_shap_plot(...)` — kembalikan grafik SHAP interpretasi fitur

### `form_utils.py`

- `encode_manual_input()` — encoding untuk form manual input

## 🤝 Acknowledgement

- Kementerian Desa, Pembangunan Daerah Tertinggal dan Transmigrasi (Kemendesa PDTT) atas data yang digunakan
- Dosen Pembimbing dan Departemen Informatika
- Pengembang pustaka open-source seperti Streamlit, Scikit-learn, SHAP, dan lainnya

## 🎓 Kontak Penulis

- **Nama:** Muhammad Rizqi
- **NIM:** 123190083
- **Program Studi:** Informatika S1
- **Universitas:** UPN "Veteran" Yogyakarta

---

## ⚖️ Lisensi

Repositori ini dibuat untuk kepentingan tugas akhir dan bersifat terbuka untuk tujuan edukasi. Mohon cantumkan referensi jika digunakan.

---

## 🌐 Referensi

- Kementerian Desa PDTT. (2023). Dataset Intervensi Stunting.
- Scikit-Learn Documentation
- SHAP Explainability
- Streamlit Documentation
- Géron, A. (2019). _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_
- Lundberg, S. M., & Lee, S.-I. (2017). _A Unified Approach to Interpreting Model Predictions_ (SHAP)

Untuk pertanyaan atau kolaborasi, silakan hubungi melalui GitHub Issues atau kontak penulis skripsi.
