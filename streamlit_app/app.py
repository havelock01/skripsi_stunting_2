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
from encoder_utils import normalize_kategorikal
from shap_utils import generate_shap_plot
from form_utils import encode_manual_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import tempfile

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
# st.write("Kolom tersedia:", df.columns.tolist())

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

# --- Cek Distribusi Label
st.markdown("---")
st.subheader("📊 Distribusi Label Efektivitas")
label_counts = df['label_efektivitas'].value_counts()
st.bar_chart(label_counts)
st.write(label_counts)

# === EVALUASI VISUAL MODEL ===
st.markdown("---")
st.subheader("📊 Evaluasi Model Random Forest")

# Encode ulang fitur untuk evaluasi model
X_eval = df[fitur_kategori].fillna("Tidak Ada")
X_eval = normalize_kategorikal(X_eval, fitur_kategori)
X_eval_encoded = X_eval.copy()
for col in fitur_kategori:
    X_eval_encoded[col] = le_dict[col].transform(X_eval[col])

# Encode label target
y_true = df['label_efektivitas']
y_encoded = LabelEncoder().fit_transform(y_true)
y_pred = model.predict(X_eval_encoded)

# Tampilkan Confusion Matrix
fig_cm, ax_cm = plt.subplots()
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Efektif", "Efektif"])
disp.plot(ax=ax_cm, cmap="Blues")
st.pyplot(fig_cm)

# --- Normalized Confusion Matrix RF
st.markdown("#### 🔍 Confusion Matrix (Normalized - RF)")
cm_norm = confusion_matrix(y_encoded, y_pred, normalize='true')
fig_cm_norm, ax_cm_norm = plt.subplots()
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Tidak Efektif", "Efektif"])
disp_norm.plot(ax=ax_cm_norm, cmap="Blues", values_format=".2f")
st.pyplot(fig_cm_norm)

# Tampilkan Classification Report
report = classification_report(y_encoded, y_pred, target_names=["Tidak Efektif", "Efektif"], output_dict=True, zero_division=0)
st.markdown("### Metrik Klasifikasi")
st.dataframe(pd.DataFrame(report).transpose().round(2))

# --- Feature Importance RF
st.markdown("---")
st.subheader("🔍 Feature Importance (Random Forest)")
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Fitur': fitur_kategori,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)

# === PERBANDINGAN MODEL: Decision Tree vs Random Forest ===
# Load model Decision Tree (karena model Random Forest sudah aktif di variable 'model')
dt_model = joblib.load("model/decision_tree_model.pkl")
# === EVALUASI MODEL DECISION TREE ===
st.subheader("📊 Evaluasi Model Decision Tree")

# Prediksi ulang
y_pred_dt_eval = dt_model.predict(X_eval_encoded)

# Confusion Matrix
fig_cm_dt, ax_cm_dt = plt.subplots()
cm_dt = confusion_matrix(y_encoded, y_pred_dt_eval)
ConfusionMatrixDisplay(cm_dt, display_labels=["Tidak Efektif", "Efektif"]).plot(ax=ax_cm_dt, cmap="Purples")
st.pyplot(fig_cm_dt)

# --- Normalized Confusion Matrix DT
st.markdown("#### 🔍 Confusion Matrix (Normalized - DT)")
cm_dt_norm = confusion_matrix(y_encoded, y_pred_dt_eval, normalize='true')
fig_cm_dt_norm, ax_cm_dt_norm = plt.subplots()
disp_dt_norm = ConfusionMatrixDisplay(confusion_matrix=cm_dt_norm, display_labels=["Tidak Efektif", "Efektif"])
disp_dt_norm.plot(ax=ax_cm_dt_norm, cmap="Purples", values_format=".2f")
st.pyplot(fig_cm_dt_norm)

# Classification Report
report_dt_eval = classification_report(y_encoded, y_pred_dt_eval, target_names=["Tidak Efektif", "Efektif"], output_dict=True, zero_division=0)
st.markdown("### Metrik Klasifikasi Decision Tree")
st.dataframe(pd.DataFrame(report_dt_eval).transpose().round(2))

# --- Tombol Download Evaluasi DT ke Excel
if st.button("⬇️ Download Evaluasi Decision Tree ke Excel"):
    report_dt_df = pd.DataFrame(report_dt_eval).transpose().round(2)
    towrite_dt = io.BytesIO()
    with pd.ExcelWriter(towrite_dt, engine='xlsxwriter') as writer:
        report_dt_df.to_excel(writer, sheet_name="Evaluasi DT")
        pd.DataFrame(cm_dt).to_excel(writer, sheet_name="Confusion Matrix")
    towrite_dt.seek(0)
    st.download_button(
        label="📄 Klik untuk Unduh Evaluasi DT",
        data=towrite_dt,
        file_name="evaluasi_decision_tree.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Prediksi ulang untuk perbandingan
y_pred_dt = dt_model.predict(X_eval_encoded)
y_pred_rf = model.predict(X_eval_encoded)

# Hitung classification report
report_dt = classification_report(y_encoded, y_pred_dt, output_dict=True, zero_division=0)
report_rf = classification_report(y_encoded, y_pred_rf, output_dict=True, zero_division=0)

# Gabungkan f1-score per kelas
df_compare = pd.DataFrame({
    "Decision Tree": pd.DataFrame(report_dt).transpose().round(2)["f1-score"],
    "Random Forest": pd.DataFrame(report_rf).transpose().round(2)["f1-score"]
})

# Tampilkan
st.markdown("---")
st.subheader("📊 Perbandingan F1-Score Decision Tree vs Random Forest")
st.dataframe(df_compare)

# Plot perbandingan f1-score
st.markdown("### 📈 Visualisasi Perbandingan F1-Score")
fig, ax = plt.subplots(figsize=(10, 4))
df_compare.plot(kind='bar', ax=ax)
ax.set_ylabel("F1-Score")
ax.set_title("Perbandingan F1-Score per Kelas")
st.pyplot(fig)

# --- Distribusi Probabilitas Prediksi RF
st.markdown("---")
st.subheader("📉 Distribusi Probabilitas Prediksi (RF)")
probas = model.predict_proba(X_eval_encoded)
fig_prob, ax_prob = plt.subplots()
ax_prob.hist(probas[:, 1], bins=20, color='skyblue')
ax_prob.set_title("Distribusi Probabilitas Prediksi - Kelas Efektif")
ax_prob.set_xlabel("Probabilitas Prediksi Efektif")
ax_prob.set_ylabel("Jumlah Desa")
st.pyplot(fig_prob)


if st.button("🖨️ Export Evaluasi ke PDF"):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simpan confusion matrix sebagai gambar
        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Tidak Efektif", "Efektif"]).plot(ax=cm_ax, cmap='Blues')
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        cm_fig.savefig(cm_path, bbox_inches='tight')

        # Simpan bar chart perbandingan
        bar_fig, bar_ax = plt.subplots(figsize=(10, 4))
        df_compare.plot(kind='bar', ax=bar_ax)
        bar_ax.set_ylabel("F1-Score")
        bar_ax.set_title("Perbandingan F1-Score DT vs RF")
        bar_path = os.path.join(tmpdir, "f1_compare.png")
        bar_fig.savefig(bar_path, bbox_inches='tight')

        # Buat PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Evaluasi Model Random Forest", ln=True, align="C")
        pdf.image(cm_path, x=10, y=30, w=180)
        pdf.ln(95)

        pdf.cell(200, 10, txt="Perbandingan F1-Score DT vs RF", ln=True, align="C")
        pdf.image(bar_path, x=10, y=135, w=180)

        pdf_path = os.path.join(tmpdir, "evaluasi_stunting.pdf")
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Unduh Evaluasi PDF",
                data=f,
                file_name="evaluasi_stunting.pdf",
                mime="application/pdf"
            )

# Tombol unduh
st.markdown("---")
if st.button("⬇️ Download Perbandingan ke Excel"):
    towrite2 = io.BytesIO()
    with pd.ExcelWriter(towrite2, engine='xlsxwriter') as writer:
        df_compare.to_excel(writer, sheet_name="F1_Perbandingan")
    towrite2.seek(0)
    st.download_button(
        label="📄 Klik untuk Unduh Perbandingan",
        data=towrite2,
        file_name="perbandingan_model_dt_rf.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



# Export Hasil Evaluasi ke Excel
st.markdown("---")
if st.button("⬇️ Download Evaluasi ke Excel"):
    report_df = pd.DataFrame(report).transpose().round(2)

    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        report_df.to_excel(writer, sheet_name="Evaluasi RF")
        pd.DataFrame(cm).to_excel(writer, sheet_name="Confusion Matrix")

    towrite.seek(0)
    st.download_button(
        label="📄 Klik untuk Unduh Evaluasi",
        data=towrite,
        file_name="evaluasi_random_forest.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Export ke Excel
st.markdown("---")
if st.button("⬇️ Download Hasil Prediksi ke Excel"):
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