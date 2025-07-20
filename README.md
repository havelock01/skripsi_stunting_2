# 🎓 Klasifikasi Efektivitas Intervensi Stunting di Desa

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-orange)](https://scikit-learn.org)

## 📖 Tentang Skripsi

**Judul:** Klasifikasi Efektivitas Respon Intervensi Stunting di Desa Menggunakan Algoritma Decision Tree dan Random Forest (Studi Kasus: Data Kementerian Desa PDTT 2023)

**Mahasiswa:** Muhammad Rizqi (123190083)  
**Program Studi:** Informatika S1  
**Universitas:** UPN "Veteran" Yogyakarta  
**Tahun:** 2025

---

## 📊 Deskripsi Project

Aplikasi machine learning berbasis web untuk mengklasifikasikan efektivitas intervensi stunting di desa menggunakan data layanan desa tahun 2023 dari Kementerian Desa PDTT. Sistem ini mengimplementasikan algoritma **Decision Tree** dan **Random Forest** dengan interpretabilitas model menggunakan **SHAP**.

### 🎯 Tujuan

- Mengklasifikasikan efektivitas intervensi stunting berdasarkan 5 indikator kunci
- Membandingkan performa algoritma Decision Tree dan Random Forest
- Menyediakan interpretasi model yang dapat dipahami stakeholder
- Membantu pengambilan keputusan dalam program pencegahan stunting

### 📈 Fitur Indikator

1. **Monitoring/Evaluasi** - Pelaksanaan monitoring konvergensi stunting
2. **Posyandu** - Aktivitas rutin penyelenggaraan posyandu
3. **RDS/TPPS** - Pembentukan Relawan Desa Sehat/Tim Percepatan Pencegahan Stunting
4. **Peningkatan Kapasitas** - Pelatihan untuk kader dan pelaku desa
5. **Ketahanan Pangan** - Program pengembangan ketahanan pangan keluarga

---

## 📂 Struktur Project

```
skripsi_stunting/
├── README.md                    # Dokumentasi project
├── requirements.txt             # Dependencies Python
├── main.py                      # Entry point preprocessing
├── test_env.py                  # Environment checker
│
├── data/                        # Dataset
│   ├── jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx
│   └── stunting_2023_labeled.csv                    # Generated
│
├── model/                       # Model hasil training
│   ├── decision_tree_model.pkl                      # Generated
│   ├── random_forest_model.pkl                      # Generated
│   ├── feature_label_encoders.pkl                   # Generated
│   └── target_label_encoder.pkl                     # Generated
│
├── src/                         # Core modules
│   ├── preprocessing.py         # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering & labeling
│   ├── model_training.py        # Model training & evaluation
│   ├── encoder_utils.py         # Encoding utilities
│   ├── evaluation.py            # Model evaluation functions
│   └── shap_utils.py            # SHAP interpretability
│
├── streamlit_app/               # Web interface
│   └── app.py                   # Main Streamlit application
│
└── notebooks/                   # Analysis notebooks (optional)
    └── eksplorasi_data.ipynb    # Data exploration
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. **Persiapan Environment**

```bash
# Clone atau download project
cd skripsi_stunting

# Buat virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. **Cek Environment**

```bash
python test_env.py
```

### 3. **Preprocessing Data**

```bash
# Pastikan file Excel ada di folder data/
python main.py
```

**Output:** `data/stunting_2023_labeled.csv`

### 4. **Training Model**

```bash
python src/model_training.py
```

**Output:** Model files di folder `model/`

### 5. **Jalankan Aplikasi Web**

```bash
streamlit run streamlit_app/app.py
```

**Akses:** `http://localhost:8501`

---

## 🔧 Kebutuhan Sistem

### Software Requirements

- **Python:** 3.8 atau lebih tinggi
- **RAM:** Minimum 4GB (Recommended 8GB)
- **Storage:** 500MB untuk dependencies + data

### Key Dependencies

- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `streamlit` - Web framework
- `shap` - Model interpretability
- `imbalanced-learn` - Handle class imbalance
- `matplotlib/seaborn` - Visualization

---

## 🧠 Metodologi

### 1. **Data Preprocessing**

- Cleaning data Excel multi-header
- Normalisasi nilai kategorikal
- Handle missing values
- Feature extraction lokasi desa

### 2. **Feature Engineering**

- Perhitungan skor efektivitas (0-5)
- Kategorisasi berdasarkan skor + anggaran:
  - **Efektif:** Skor ≥ 3 + ada anggaran
  - **Cukup Efektif:** Skor 1-2 + anggaran ≥ median
  - **Kurang Efektif:** Skor 0 atau tidak ada anggaran

### 3. **Model Training**

- **Algorithms:** Decision Tree, Random Forest
- **Class Imbalance:** SMOTE oversampling
- **Evaluation:** Accuracy, Precision, Recall, F1-Score
- **Cross Validation:** 5-fold

### 4. **Model Interpretability**

- **SHAP:** Feature importance & individual predictions
- **Feature Importance:** Built-in model importance
- **Confusion Matrix:** Performance visualization

---

## 📊 Fitur Aplikasi

### 🎯 **Dashboard Utama**

- Ringkasan dataset dan distribusi
- Filter data berdasarkan kabupaten
- Visualisasi distribusi label dan skor

### 🔮 **Prediksi Manual**

- Form input untuk 5 indikator
- Prediksi real-time dengan kedua model
- Interpretasi SHAP untuk setiap prediksi

### 📈 **Evaluasi Model**

- Confusion Matrix untuk kedua model
- Classification Report detail
- Perbandingan performa RF vs DT
- Feature importance analysis

### 📤 **Export & Download**

- Download data terfilter (CSV)
- Export evaluasi model (Excel)
- Visualisasi hasil analisis

---

## 🎓 Kontribusi Ilmiah

### **Novelty**

1. **Dataset Terbaru:** Menggunakan data Kemendes PDTT 2023
2. **Multi-Indikator:** Kombinasi 5 indikator kunci stunting
3. **Interpretability:** SHAP untuk transparansi model
4. **Practical Application:** Web interface untuk stakeholder

### **Expected Results**

- Akurasi model > 85%
- Identifikasi fitur paling berpengaruh
- Rekomendasi kebijakan berbasis data
- Framework untuk replikasi di daerah lain

---

## 📝 Troubleshooting

### **Error: File tidak ditemukan**

```bash
# Pastikan file Excel ada di folder data/
ls data/jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx

# Jalankan preprocessing ulang
python main.py
```

### **Error: Model tidak ditemukan**

```bash
# Jalankan training ulang
python src/model_training.py

# Cek file model
ls model/*.pkl
```

### **Error: Import module**

```bash
# Pastikan di root directory
pwd  # Harus di skripsi_stunting/

# Reinstall dependencies
pip install -r requirements.txt
```

### **Error: Streamlit tidak bisa akses model**

```bash
# Jalankan dari root directory
cd skripsi_stunting
streamlit run streamlit_app/app.py
```

---

## 📚 Referensi

### **Academic References**

1. Kementerian Desa PDTT. (2023). Dataset Layanan Pencegahan Stunting
2. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions
4. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Oversampling Technique

### **Technical Documentation**

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 🤝 Acknowledgments

- **Kementerian Desa PDTT** - Penyedia dataset
- **Dosen Pembimbing: Bagus Muhammad Akbar, S.ST., M.Kom.** - Guidance dan supervisi
- **UPN "Veteran" Yogyakarta** - Institusi pendukung
- **Open Source Community** - Libraries dan tools

---

## 📧 Kontak

**Muhammad Rizqi**  
📧 Email: [mrizqi0153@gmail.com]  
🎓 NIM: 123190083  
🏫 Informatika S1 - UPN "Veteran" Yogyakarta

---

## ⚖️ Lisensi

Project ini dibuat untuk keperluan akademik (skripsi). Silakan gunakan untuk tujuan edukasi dengan menyertakan referensi yang sesuai.

**Citation:**

```
Rizqi, M. (2025). Klasifikasi Efektivitas Respon Intervensi Stunting di Desa
Menggunakan Algoritma Decision Tree dan Random Forest.
Skripsi, Informatika S1, UPN "Veteran" Yogyakarta.
```

---
