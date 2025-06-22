import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import io
from datetime import datetime

from encoder_utils import fit_label_encoders, normalize_kategorikal
from shap_utils import generate_shap_plot
from form_utils import encode_manual_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
import tempfile

# --- Config
st.set_page_config(page_title="Klasifikasi Efektivitas Intervensi", layout="wide")
st.title("Klasifikasi Efektivitas Intervensi Stunting di Desa")

# --- Load model & data
model_rf = joblib.load("model/random_forest_model.pkl")
model_dt = joblib.load("model/decision_tree_model.pkl")
df = pd.read_csv("data/stunting_2023_labeled.csv")

# --- Fit LabelEncoder untuk target
y_label = df['label_efektivitas']
le_label = LabelEncoder().fit(y_label)
labels = le_label.classes_.tolist()

# --- Fitur kategori indikator
dummy_features = [
    'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
    'Aktivitas_rutin_Penyelenggaraan_posyandu_',
    'Terdapat_Pembentukan_RDS/TPPS',
    'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
    'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
]

# Fungsi untuk encoding fitur
def encode_features(df, le_dict, features):
    df_enc = df[features].fillna("Tidak Ada").apply(lambda col: col.str.strip().str.lower())
    for col in features:
        # Mapping nilai yang tidak ada di encoder ke kelas pertama
        df_enc[col] = df_enc[col].where(df_enc[col].isin(le_dict[col].classes_), le_dict[col].classes_[0])
        df_enc[col] = le_dict[col].transform(df_enc[col])
    return df_enc

# --- Encode seluruh dataset untuk prediksi otomatis
X_raw, le_dict = fit_label_encoders(df[dummy_features].copy(), dummy_features)
pred_enc = model_rf.predict(X_raw)
df['Prediksi_Model'] = le_label.inverse_transform(pred_enc)

# --- Sidebar Filter
st.sidebar.header("Filter Data")
if 'NAMA_KABUPATEN' in df.columns:
    choices = df['NAMA_KABUPATEN'].dropna().unique().tolist()
    sel_kab = st.sidebar.multiselect("Pilih Kabupaten", options=choices)
    if sel_kab:
        df = df[df['NAMA_KABUPATEN'].isin(sel_kab)]

# --- Tampilkan Data & Prediksi
st.subheader("Data dan Prediksi")
display_cols = ['NAMA_DESA'] + dummy_features + ['Prediksi_Model']
st.dataframe(df[display_cols].reset_index(drop=True))

# --- SHAP Global (Sidebar)
with st.sidebar.expander("Interpretasi SHAP (Global)", expanded=False):
    X_disp = df[dummy_features].fillna("Tidak Ada")
    for col in dummy_features:
        X_disp[col] = X_disp[col].where(X_disp[col].isin(le_dict[col].classes_), le_dict[col].classes_[0])
    X_disp_enc = pd.DataFrame({col: le_dict[col].transform(X_disp[col]) for col in dummy_features})
    explainer = shap.TreeExplainer(model_rf)
    shap_values = explainer.shap_values(X_disp_enc)
    st.markdown("#### SHAP Summary Plot (Global)")
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_disp_enc, feature_names=dummy_features, show=False)
    st.pyplot(fig_shap)

# --- SHAP Instance (Main Panel)
st.sidebar.markdown("---")
selected_class = st.sidebar.radio("Interpretasi SHAP untuk kelas:", labels)
selected_idx = labels.index(selected_class)

st.subheader("Interpretasi SHAP")
idx = st.number_input("Pilih index desa (0-n)", min_value=0, max_value=len(df)-1, value=0)
st.markdown(f"**Nama Desa:** {df.iloc[idx]['NAMA_DESA']}")

fig_shap_instance, _, _ = generate_shap_plot(model_rf, X_disp_enc, idx, selected_idx, dummy_features)
st.pyplot(fig_shap_instance)

# --- Manual Input Form (dropdown seimbang & label singkat)
st.subheader("Prediksi Manual")
col_count = len(dummy_features)
cols = st.columns(col_count, gap="medium")  # gunakan gap agar lebih rapi
manual_input = {}

# Label singkat untuk tampilan
label_map = {
    dummy_features[0]: "Monitoring/Evaluasi",
    dummy_features[1]: "Posyandu",
    dummy_features[2]: "RDS/TPPS",
    dummy_features[3]: "Peningkatan Kapasitas",
    dummy_features[4]: "Ketahanan Pangan"
}

with st.form("manual_form"):
    for i, col in enumerate(dummy_features):
        opsi = sorted(df[col].dropna().unique().tolist())
        manual_input[col] = cols[i].selectbox(label_map.get(col, col), opsi, key=f"manual_{col}")
    submit = st.form_submit_button("Prediksi Efektivitas")

if submit:
    df_enc = pd.DataFrame([manual_input])
    # Normalisasi manual input agar sama dengan training
    df_enc = df_enc.apply(lambda col: col.str.strip().str.lower())
    # Pastikan semua nilai ada di encoder, jika tidak mapping ke kelas pertama
    for col in dummy_features:
        df_enc[col] = df_enc[col].where(df_enc[col].isin(le_dict[col].classes_), le_dict[col].classes_[0])
    df_enc = pd.DataFrame({col: le_dict[col].transform(df_enc[col]) for col in dummy_features})

    st.markdown("🔍 **Hasil encoding input manual:**")
    st.dataframe(df_enc)

    st.markdown("#### Mapping LabelEncoder fitur:")
    for col in dummy_features:
        le = le_dict[col]
        st.text(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    pred_m = model_rf.predict(df_enc)[0]
    label_m = le_label.inverse_transform([pred_m])[0]
    st.success(f"Prediksi: **{label_m}**")

    expl = shap.TreeExplainer(model_rf).shap_values(df_enc)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(
        values=expl[pred_m][0],
        base_values=shap.TreeExplainer(model_rf).expected_value[pred_m],
        data=df_enc.iloc[0],
        feature_names=dummy_features
    ), show=False)
    st.pyplot(fig)

# === Tabs Evaluasi ===
tab1, tab2, tab3, tab4 = st.tabs(["📉 Random Forest", "🌳 Decision Tree", "⚖️ Perbandingan", "📤 Unduhan"])

with tab1:
    st.subheader("📊 Evaluasi Model Random Forest")
    X_eval = encode_features(df, le_dict, dummy_features)
    y_true = df['label_efektivitas']
    y_enc = le_label.transform(y_true)
    y_pred = model_rf.predict(X_eval)

    # Ambil label unik setelah filter
    unique_labels_rf = sorted(list(set(y_enc) | set(y_pred)))
    display_labels_rf = [labels[i] for i in unique_labels_rf]

    # Confusion Matrix
    cm = confusion_matrix(y_enc, y_pred, labels=unique_labels_rf)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_rf).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # Normalized Confusion Matrix
    st.markdown("#### Confusion Matrix (Normalized)")
    cm_norm = confusion_matrix(y_enc, y_pred, labels=unique_labels_rf, normalize='true')
    fig_n, ax_n = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=display_labels_rf).plot(ax=ax_n, values_format='.2f')
    st.pyplot(fig_n)

    # Classification Report
    report_rf = classification_report(y_enc, y_pred, labels=unique_labels_rf, target_names=display_labels_rf, output_dict=True, zero_division=0)
    st.markdown("### Metrik Klasifikasi RF")
    st.dataframe(pd.DataFrame(report_rf).transpose().round(2))

    # Feature Importance
    st.markdown("### 🔍 Feature Importance RF")
    feat_imp = pd.DataFrame({
        "Fitur": dummy_features,
        "Importance": model_rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.dataframe(feat_imp)

with tab2:
    st.subheader("📊 Evaluasi Model Decision Tree")
    y_pred_dt = model_dt.predict(X_eval)
    unique_labels_dt = sorted(list(set(y_enc) | set(y_pred_dt)))
    display_labels_dt = [labels[i] for i in unique_labels_dt]

    fig_dt, ax_dt = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_enc, y_pred_dt, labels=unique_labels_dt),
        display_labels=display_labels_dt
    ).plot(ax=ax_dt)
    st.pyplot(fig_dt)

    st.markdown("#### Confusion Matrix DT (Normalized)")
    fig_dt_n, ax_dt_n = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_enc, y_pred_dt, labels=unique_labels_dt, normalize='true'),
        display_labels=display_labels_dt
    ).plot(ax=ax_dt_n, values_format='.2f')
    st.pyplot(fig_dt_n)

    report_dt = classification_report(y_enc, y_pred_dt, labels=unique_labels_dt, target_names=display_labels_dt, output_dict=True, zero_division=0)
    st.markdown("### Metrik Klasifikasi DT")
    st.dataframe(pd.DataFrame(report_dt).transpose().round(2))

with tab3:
    st.subheader("📊 Perbandingan F1-Score DT vs RF")
    f1_rf = pd.DataFrame(report_rf).transpose()['f1-score']
    f1_dt = pd.DataFrame(report_dt).transpose()['f1-score']
    df_comp = pd.DataFrame({'Random Forest': f1_rf, 'Decision Tree': f1_dt})
    st.dataframe(df_comp)

    fig_comp, ax_comp = plt.subplots(figsize=(10,4))
    df_comp.plot(kind='bar', ax=ax_comp)
    ax_comp.set_ylabel('F1-Score')
    ax_comp.set_title('Perbandingan F1-Score per Kelas')
    st.pyplot(fig_comp)

with tab4:
    st.subheader("📤 Unduhan Evaluasi & Hasil Prediksi")
    # Download Evaluasi RF
    if st.button("⬇️ Download Evaluasi Random Forest ke Excel"):
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            pd.DataFrame(report_rf).transpose().round(2).to_excel(writer, sheet_name="Evaluasi_RF")
            pd.DataFrame(cm, index=display_labels_rf, columns=display_labels_rf).to_excel(writer, sheet_name="Confusion_Matrix_RF")
        towrite.seek(0)
        st.download_button("📄 Unduh Evaluasi RF", data=towrite, file_name="evaluasi_rf.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Download Evaluasi DT
    if st.button("⬇️ Download Evaluasi Decision Tree ke Excel"):
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
            pd.DataFrame(report_dt).transpose().round(2).to_excel(writer, sheet_name="Evaluasi_DT")
            pd.DataFrame(confusion_matrix(y_enc, y_pred_dt, labels=unique_labels_dt), index=display_labels_dt, columns=display_labels_dt).to_excel(writer, sheet_name="Confusion_Matrix_DT")
        towrite.seek(0)
        st.download_button("📄 Unduh Evaluasi DT", data=towrite, file_name="evaluasi_dt.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Download Perbandingan F1
    if st.button("⬇️ Download Perbandingan ke Excel"):
        towrite_cmp = io.BytesIO()
        with pd.ExcelWriter(towrite_cmp, engine='xlsxwriter') as writer:
            df_comp.to_excel(writer, sheet_name="Perbandingan_F1")
        towrite_cmp.seek(0)
        st.download_button("📄 Unduh Perbandingan F1", data=towrite_cmp, file_name="perbandingan_f1.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Export PDF
    if st.button("🖨️ Export Evaluasi ke PDF"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confusion Matrix Image
            cm_fig, cm_ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix(y_enc, y_pred), display_labels=labels).plot(ax=cm_ax)
            cm_path = os.path.join(tmpdir, "cm.png")
            cm_fig.savefig(cm_path, bbox_inches='tight')

            # F1 Comparison Image
            comp_fig, comp_ax = plt.subplots(figsize=(10,4))
            df_comp.plot(kind='bar', ax=comp_ax)
            comp_ax.set_ylabel('F1-Score')
            comp_ax.set_title('Perbandingan F1-Score DT vs RF')
            comp_path = os.path.join(tmpdir, "comp.png")
            comp_fig.savefig(comp_path, bbox_inches='tight')

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Evaluasi Model Random Forest", ln=True, align='C')
            pdf.image(cm_path, x=10, y=25, w=180)
            pdf.ln(95)
            pdf.cell(0, 10, txt="Perbandingan F1-Score DT vs RF", ln=True, align='C')
            pdf.image(comp_path, x=10, y=130, w=180)

            pdf_path = os.path.join(tmpdir, "evaluasi.pdf")
            pdf.output(pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📄 Unduh Evaluasi PDF", data=f, file_name="evaluasi.pdf", mime="application/pdf")
                
st.write("Distribusi label (full):", df['label_efektivitas'].value_counts())
