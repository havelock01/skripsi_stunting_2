import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import io
from datetime import datetime

from encoder_utils import fit_label_encoders
from shap_utils import generate_shap_plot
from form_utils import encode_manual_input

st.set_page_config(page_title="Klasifikasi Efektivitas Intervensi", layout="wide")

st.title("Klasifikasi Efektivitas Intervensi Stunting di Desa")

model = joblib.load("model/random_forest_model.pkl")
df = pd.read_csv("data/stunting_2023_labeled.csv")

fitur_kategori = [
    'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
    'Aktivitas_rutin_Penyelenggaraan_posyandu_',
    'Terdapat_Pembentukan_RDS/TPPS',
    'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
    'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
]

# --- Encode awal seluruh dataset
X_raw, le_dict = fit_label_encoders(df[fitur_kategori].copy(), fitur_kategori)
pred = model.predict(X_raw)
df['Prediksi_Model'] = ['Efektif' if p == 1 else 'Tidak Efektif' for p in pred]

# --- Sidebar Filter
st.sidebar.header("Filter Data")
if 'NAMA_KABUPATEN' in df.columns:
    kabupaten = st.sidebar.multiselect("Pilih Kabupaten", options=df['NAMA_KABUPATEN'].dropna().unique())
    if kabupaten:
        df = df[df['NAMA_KABUPATEN'].isin(kabupaten)]

# --- Data Display
st.subheader("Data dan Prediksi")
df_display = df[['NAMA_DESA'] + fitur_kategori + ['Prediksi_Model']].reset_index(drop=True)
st.dataframe(df_display)

# --- SHAP Class Selection
st.sidebar.markdown("---")
selected_class_label = st.sidebar.radio("Interpretasi SHAP untuk kelas:", ["Efektif", "Tidak Efektif"])
selected_class_index = 1 if selected_class_label == "Efektif" else 0

# --- SHAP Input & Nama Desa
st.subheader("Interpretasi SHAP")
selected_index = st.number_input("Pilih index desa (0-n)", min_value=0, max_value=len(df_display)-1, value=0)
st.markdown(f"**Nama Desa:** {df_display.iloc[selected_index]['NAMA_DESA']}")

# --- Generate SHAP plot dari data display
X_encoded_display, _ = fit_label_encoders(df_display[fitur_kategori].copy(), fitur_kategori)
fig, shap_values, explainer = generate_shap_plot(model, X_encoded_display, selected_index, selected_class_index, fitur_kategori)
st.pyplot(fig)

# --- Form Manual Input
st.subheader("Prediksi Manual")
with st.form("manual_form"):
    input_data = {}
    for col in fitur_kategori:
        input_data[col] = st.selectbox(col, ["Ada", "Tidak Ada", "Rutin Tiap Bulan"], key=col)
    submitted = st.form_submit_button("Prediksi Efektivitas")

if submitted:
    df_encoded = encode_manual_input(input_data, le_dict, fitur_kategori)

    # 🔍 Tampilkan hasil encoding
    st.markdown("🔍 **Hasil encoding input manual:**")
    st.dataframe(df_encoded)

    # 📋 Tampilkan mapping encoder
    st.markdown("#### Mapping LabelEncoder:")
    for col in fitur_kategori:
        le = le_dict[col]
        st.text(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 🔮 Prediksi dan SHAP
    pred_manual = model.predict(df_encoded)[0]
    label_pred = "Efektif" if pred_manual == 1 else "Tidak Efektif"
    st.success(f"Prediksi: **{label_pred}**")

    expl_manual = shap.TreeExplainer(model).shap_values(df_encoded)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=expl_manual[pred_manual][0],
        base_values=shap.TreeExplainer(model).expected_value[pred_manual],
        data=df_encoded.iloc[0],
        feature_names=fitur_kategori
    ), show=False)
    st.pyplot(fig)


# --- Visualisasi Tambahan
st.subheader("Ringkasan Prediksi")
pie_df = df['Prediksi_Model'].value_counts().reset_index()
pie_df.columns = ['Efektivitas', 'Jumlah']
fig_pie = px.pie(pie_df, names='Efektivitas', values='Jumlah', title='Distribusi Prediksi')
st.plotly_chart(fig_pie)

if 'skor' in df.columns and 'NAMA_KABUPATEN' in df.columns:
    st.subheader("Rata-rata Skor per Kabupaten")
    mean_score = df.groupby('NAMA_KABUPATEN')['skor'].mean().reset_index().sort_values(by='skor', ascending=False)
    fig_bar = px.bar(mean_score, x='NAMA_KABUPATEN', y='skor', labels={'skor': 'Rata-rata Skor'})
    st.plotly_chart(fig_bar)

# --- Export ke Excel
st.markdown("---")
if st.button("Download Hasil Prediksi ke Excel"):
    export_df = df[['NAMA_KABUPATEN', 'NAMA_KECAMATAN', 'NAMA_DESA'] + fitur_kategori + ['Prediksi_Model']]
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name="Hasil Prediksi")
    towrite.seek(0)
    st.download_button(
        label="Klik untuk Unduh",
        data=towrite,
        file_name=f"hasil_prediksi_stunting_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
