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
    st.markdown("🔍 **Hasil encoding input manual:**")
    st.dataframe(df_encoded)
    st.markdown("#### Mapping LabelEncoder:")
    for col in fitur_kategori:
        le = le_dict[col]
        st.text(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

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

# === Tabs Evaluasi ===
tab1, tab2, tab3, tab4 = st.tabs(["📉 Random Forest", "🌳 Decision Tree", "⚖️ Perbandingan", "📤 Unduhan"])

with tab1:
    st.subheader("📊 Evaluasi Model Random Forest")
    X_eval = df[fitur_kategori].fillna("Tidak Ada")
    X_eval = normalize_kategorikal(X_eval, fitur_kategori)
    X_eval_encoded = X_eval.copy()
    for col in fitur_kategori:
        X_eval_encoded[col] = le_dict[col].transform(X_eval[col])

    y_true = df['label_efektivitas']
    y_encoded = LabelEncoder().fit_transform(y_true)
    y_pred = model.predict(X_eval_encoded)

    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Efektif", "Efektif"])
    disp.plot(ax=ax_cm, cmap="Blues")
    st.pyplot(fig_cm)

    st.markdown("#### 🔍 Confusion Matrix (Normalized - RF)")
    cm_norm = confusion_matrix(y_encoded, y_pred, normalize='true')
    fig_cm_norm, ax_cm_norm = plt.subplots()
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Tidak Efektif", "Efektif"])
    disp_norm.plot(ax=ax_cm_norm, cmap="Blues", values_format=".2f")
    st.pyplot(fig_cm_norm)

    report = classification_report(y_encoded, y_pred, target_names=["Tidak Efektif", "Efektif"], output_dict=True, zero_division=0)
    st.markdown("### Metrik Klasifikasi")
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.markdown("---")
    st.subheader("🔍 Feature Importance (Random Forest)")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Fitur': fitur_kategori, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)

with tab2:
    st.subheader("📊 Evaluasi Model Decision Tree")
    dt_model = joblib.load("model/decision_tree_model.pkl")
    y_pred_dt_eval = dt_model.predict(X_eval_encoded)

    fig_cm_dt, ax_cm_dt = plt.subplots()
    cm_dt = confusion_matrix(y_encoded, y_pred_dt_eval)
    ConfusionMatrixDisplay(cm_dt, display_labels=["Tidak Efektif", "Efektif"]).plot(ax=ax_cm_dt, cmap="Purples")
    st.pyplot(fig_cm_dt)

    st.markdown("#### 🔍 Confusion Matrix (Normalized - DT)")
    cm_dt_norm = confusion_matrix(y_encoded, y_pred_dt_eval, normalize='true')
    fig_cm_dt_norm, ax_cm_dt_norm = plt.subplots()
    disp_dt_norm = ConfusionMatrixDisplay(confusion_matrix=cm_dt_norm, display_labels=["Tidak Efektif", "Efektif"])
    disp_dt_norm.plot(ax=ax_cm_dt_norm, cmap="Purples", values_format=".2f")
    st.pyplot(fig_cm_dt_norm)

    report_dt_eval = classification_report(y_encoded, y_pred_dt_eval, target_names=["Tidak Efektif", "Efektif"], output_dict=True, zero_division=0)
    st.markdown("### Metrik Klasifikasi Decision Tree")
    st.dataframe(pd.DataFrame(report_dt_eval).transpose().round(2))

with tab3:
    st.subheader("📊 Perbandingan F1-Score Decision Tree vs Random Forest")
    report_dt = classification_report(y_encoded, y_pred_dt_eval, output_dict=True, zero_division=0)
    report_rf = classification_report(y_encoded, y_pred, output_dict=True, zero_division=0)

    df_compare = pd.DataFrame({
        "Decision Tree": pd.DataFrame(report_dt).transpose().round(2)["f1-score"],
        "Random Forest": pd.DataFrame(report_rf).transpose().round(2)["f1-score"]
    })
    st.dataframe(df_compare)

    st.markdown("### 📈 Visualisasi Perbandingan F1-Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    df_compare.plot(kind='bar', ax=ax)
    ax.set_ylabel("F1-Score")
    ax.set_title("Perbandingan F1-Score per Kelas")
    st.pyplot(fig)

with tab4:
    st.subheader("📤 Unduhan Evaluasi & Hasil Prediksi")
    if st.button("⬇️ Download Evaluasi Random Forest ke Excel"):
        report_df = pd.DataFrame(report).transpose().round(2)
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, sheet_name="Evaluasi RF")
            pd.DataFrame(cm).to_excel(writer, sheet_name="Confusion Matrix")
        towrite.seek(0)
        st.download_button(
            label="📄 Klik untuk Unduh Evaluasi RF",
            data=towrite,
            file_name="evaluasi_random_forest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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

    if st.button("🖨️ Export Evaluasi ke PDF"):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_fig, cm_ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=["Tidak Efektif", "Efektif"]).plot(ax=cm_ax, cmap='Blues')
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            cm_fig.savefig(cm_path, bbox_inches='tight')

            bar_fig, bar_ax = plt.subplots(figsize=(10, 4))
            df_compare.plot(kind='bar', ax=bar_ax)
            bar_ax.set_ylabel("F1-Score")
            bar_ax.set_title("Perbandingan F1-Score DT vs RF")
            bar_path = os.path.join(tmpdir, "f1_compare.png")
            bar_fig.savefig(bar_path, bbox_inches='tight')

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
