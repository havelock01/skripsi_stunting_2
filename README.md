# Skripsi: Klasifikasi Efektivitas Intervensi Stunting di Desa

## 📖 Judul

**Klasifikasi Efektivitas Respon Intervensi Stunting di Desa Menggunakan Algoritma Decision Tree dan Random Forest (Studi Kasus: Data Kementerian Desa PDTT 2023)**

---

## 📊 Deskripsi

Proyek ini merupakan aplikasi klasifikasi berbasis _machine learning_ yang bertujuan untuk memprediksi efektivitas intervensi stunting berdasarkan data layanan desa. Dibangun menggunakan Python dan Streamlit, proyek ini mengadopsi model _Decision Tree_ dan _Random Forest_ untuk mengevaluasi keberhasilan konvergensi stunting di desa berdasarkan data tahun 2023 dari Kementerian Desa PDTT.

---

## 📂 Struktur Direktori

```
skripsi_stunting/
│   README.md
│   main.py
│   requirements.txt
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

## 🌐 Dependencies Utama

- pandas, numpy
- scikit-learn
- matplotlib, seaborn, plotly
- shap, joblib
- streamlit, fpdf

---

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
