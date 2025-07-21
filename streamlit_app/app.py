#!/usr/bin/env python3
"""
Aplikasi Streamlit untuk Klasifikasi Efektivitas Intervensi Stunting
Skripsi: Muhammad Rizqi (123190083) - Informatika S1, UPN "Veteran" Yogyakarta

Judul: Klasifikasi Efektivitas Respon Intervensi Stunting di Desa 
       Menggunakan Algoritma Decision Tree dan Random Forest
       (Studi Kasus: Data Kementerian Desa PDTT 2023)
"""

import sys
import os
from pathlib import Path

# Setup path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
src_dir = parent_dir / "src"
sys.path.insert(0, str(src_dir))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from datetime import datetime
import io

# Import modules
try:
    from encoder_utils import encode_features, normalize_kategorikal
    from shap_utils import generate_shap_plot
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Pastikan semua file di folder 'src' tersedia")
    st.stop()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Efektivitas Intervensi Stunting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_data
def load_data():
    """Load dataset dengan caching"""
    data_path = parent_dir / "data" / "stunting_2023_labeled.csv"
    if not data_path.exists():
        st.error(f"❌ File data tidak ditemukan: {data_path}")
        st.error("Jalankan `python main.py` terlebih dahulu")
        st.stop()
    return pd.read_csv(data_path)

@st.cache_resource
def load_models():
    """Load semua model dan encoder dengan caching"""
    model_dir = parent_dir / "model"
    
    required_files = {
        "decision_tree_model.pkl": "Decision Tree Model",
        "random_forest_model.pkl": "Random Forest Model", 
        "feature_label_encoders.pkl": "Feature Encoders",
        "target_label_encoder.pkl": "Target Encoder"
    }
    
    # Check file existence
    for filename, description in required_files.items():
        filepath = model_dir / filename
        if not filepath.exists():
            st.error(f"❌ {description} tidak ditemukan: {filepath}")
            st.error("Jalankan `python src/model_training.py` terlebih dahulu")
            st.stop()
    
    # Load models and encoders
    try:
        models = {
            'Decision Tree': joblib.load(model_dir / "decision_tree_model.pkl"),
            'Random Forest': joblib.load(model_dir / "random_forest_model.pkl")
        }
        
        le_dict = joblib.load(model_dir / "feature_label_encoders.pkl")
        target_le = joblib.load(model_dir / "target_label_encoder.pkl")
        
        return models, le_dict, target_le
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

def encode_dataset_for_prediction(df, le_dict, features):
    """Encode dataset untuk prediksi"""
    try:
        return encode_features(df, le_dict, features)
    except Exception as e:
        st.error(f"❌ Error encoding dataset: {e}")
        return None

# ==========================================
# KONSTANTA
# ==========================================

FEATURE_COLUMNS = [
    'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
    'Aktivitas_rutin_Penyelenggaraan_posyandu_',
    'Terdapat_Pembentukan_RDS/TPPS',
    'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
    'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
]

FEATURE_LABELS = {
    FEATURE_COLUMNS[0]: "Monitoring/Evaluasi",
    FEATURE_COLUMNS[1]: "Posyandu",
    FEATURE_COLUMNS[2]: "RDS/TPPS", 
    FEATURE_COLUMNS[3]: "Peningkatan Kapasitas",
    FEATURE_COLUMNS[4]: "Ketahanan Pangan"
}

# ==========================================
# LOAD DATA DAN MODEL
# ==========================================

# Load data dan model
df = load_data()
models, le_dict, target_le = load_models()
labels = target_le.classes_.tolist()

# Validasi kolom yang dibutuhkan
missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
if missing_features:
    st.error(f"❌ Kolom tidak ditemukan: {missing_features}")
    st.stop()

# Encode dataset untuk prediksi
X_encoded = encode_dataset_for_prediction(df, le_dict, FEATURE_COLUMNS)
if X_encoded is not None:
    rf_predictions = models['Random Forest'].predict(X_encoded)
    df['Prediksi_RF'] = target_le.inverse_transform(rf_predictions)
    
    dt_predictions = models['Decision Tree'].predict(X_encoded)  
    df['Prediksi_DT'] = target_le.inverse_transform(dt_predictions)

# ==========================================
# HEADER APLIKASI
# ==========================================

st.title("🎯 Klasifikasi Efektivitas Intervensi Stunting di Desa")
st.markdown("---")

# Info Skripsi
with st.expander("📚 Informasi Skripsi", expanded=False):
    st.markdown("""
    **Judul:** Klasifikasi Efektivitas Respon Intervensi Stunting di Desa Menggunakan Algoritma Decision Tree dan Random Forest (Studi Kasus: Data Kementerian Desa PDTT 2023)
    
    **Mahasiswa:** Muhammad Rizqi (123190083)  
    **Program Studi:** Informatika S1  
    **Universitas:** UPN "Veteran" Yogyakarta
    
    **Dataset:** desa dari data Kementerian Desa PDTT 2023  
    **Fitur:** indikator efektivitas intervensi stunting
    """)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔍 Filter Data")

# Filter berdasarkan kabupaten
df_filtered = df.copy()
if 'NAMA_KABUPATEN' in df.columns:
    kabupaten_list = ['Semua'] + sorted(df['NAMA_KABUPATEN'].dropna().unique().tolist())
    selected_kabupaten = st.sidebar.selectbox("Pilih Kabupaten:", kabupaten_list)
    
    if selected_kabupaten != 'Semua':
        df_filtered = df_filtered[df_filtered['NAMA_KABUPATEN'] == selected_kabupaten]

# Filter berdasarkan prediksi
prediksi_list = ['Semua'] + sorted(df_filtered['Prediksi_RF'].unique().tolist())
selected_prediksi = st.sidebar.selectbox("Filter Prediksi:", prediksi_list)

if selected_prediksi != 'Semua':
    df_filtered = df_filtered[df_filtered['Prediksi_RF'] == selected_prediksi]

# Info data terfilter
st.sidebar.markdown("---")
st.sidebar.metric("Total Desa", len(df))
st.sidebar.metric("Desa Terfilter", len(df_filtered))

if len(df_filtered) > 0:
    efektif_count = len(df_filtered[df_filtered['Prediksi_RF'] == 'Efektif'])
    st.sidebar.metric("Prediksi Efektif", efektif_count)

# ==========================================
# RINGKASAN DATASET
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Desa", len(df))
with col2:
    st.metric("Jumlah Fitur", len(FEATURE_COLUMNS))
with col3:
    if 'label_efektivitas' in df.columns:
        efektif_asli = len(df[df['label_efektivitas'] == 'Efektif'])
        st.metric("Label Efektif (Asli)", efektif_asli)
    else:
        st.metric("Label Efektif (Asli)", "N/A")
with col4:
    efektif_pred = len(df[df['Prediksi_RF'] == 'Efektif'])
    st.metric("Prediksi Efektif (RF)", efektif_pred)

# ==========================================
# DATA VISUALIZATION
# ==========================================

st.markdown("---")
st.subheader("📊 Visualisasi Data")

# Tabs untuk berbagai visualisasi
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Distribusi", "🗺️ Per Kabupaten", "📋 Tabel Data"])

with viz_tab1:
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Distribusi label asli vs prediksi
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'label_efektivitas' in df.columns:
            df['label_efektivitas'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%')
            ax1.set_title('Distribusi Label Asli')
        
        df['Prediksi_RF'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Distribusi Prediksi Random Forest')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_viz2:
        # Distribusi skor
        if 'skor' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['skor'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title('Distribusi Skor Efektivitas')
            ax.set_xlabel('Skor')
            ax.set_ylabel('Jumlah Desa')
            plt.xticks(rotation=0)
            st.pyplot(fig)

with viz_tab2:
    # Analisis per kabupaten
    if 'NAMA_KABUPATEN' in df.columns:
        kabupaten_summary = df.groupby('NAMA_KABUPATEN').agg({
            'Prediksi_RF': lambda x: (x == 'Efektif').sum(),
            'NAMA_DESA': 'count'
        }).rename(columns={'Prediksi_RF': 'Efektif', 'NAMA_DESA': 'Total'})
        
        kabupaten_summary['Persentase_Efektif'] = (kabupaten_summary['Efektif'] / kabupaten_summary['Total'] * 100).round(1)
        kabupaten_summary = kabupaten_summary.sort_values('Persentase_Efektif', ascending=False)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        kabupaten_summary['Persentase_Efektif'].plot(kind='barh', ax=ax)
        ax.set_title('Persentase Prediksi Efektif per Kabupaten')
        ax.set_xlabel('Persentase (%)')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tabel summary
        st.dataframe(kabupaten_summary, use_container_width=True)

with viz_tab3:
    # Tabel data dengan prediksi
    display_columns = []
    if 'NAMA_DESA' in df_filtered.columns:
        display_columns.append('NAMA_DESA')
    if 'NAMA_KABUPATEN' in df_filtered.columns:
        display_columns.append('NAMA_KABUPATEN')
    
    display_columns.extend(FEATURE_COLUMNS)
    
    if 'label_efektivitas' in df_filtered.columns:
        display_columns.extend(['label_efektivitas', 'Prediksi_RF', 'Prediksi_DT'])
    else:
        display_columns.extend(['Prediksi_RF', 'Prediksi_DT'])
    
    if 'skor' in df_filtered.columns:
        display_columns.append('skor')
    
    # Filter kolom yang benar-benar ada
    available_columns = [col for col in display_columns if col in df_filtered.columns]
    
    st.dataframe(
        df_filtered[available_columns].reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv_data = df_filtered[available_columns].to_csv(index=False)
    st.download_button(
        label="📥 Download Data Terfilter (CSV)",
        data=csv_data,
        file_name=f"data_stunting_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ==========================================
# PREDIKSI MANUAL
# ==========================================

st.markdown("---")
st.subheader("🔮 Prediksi Manual")

with st.form("manual_prediction_form"):
    st.markdown("**Pilih kondisi untuk setiap indikator:**")
    
    cols = st.columns(len(FEATURE_COLUMNS))
    manual_input = {}
    
    for i, feature in enumerate(FEATURE_COLUMNS):
        options = ["Tidak Ada", "Ada", "Rutin Tiap Bulan"]
        manual_input[feature] = cols[i].selectbox(
            FEATURE_LABELS[feature],
            options,
            key=f"manual_{i}"
        )
    
    col_submit1, col_submit2 = st.columns([1, 4])
    with col_submit1:
        submit_manual = st.form_submit_button("🎯 Prediksi", use_container_width=True)
    
    if submit_manual:
        try:
            # Encode input manual
            df_manual = pd.DataFrame([manual_input])
            df_manual_encoded = encode_features(df_manual, le_dict, FEATURE_COLUMNS)
            
            # Prediksi dengan kedua model
            pred_rf = models['Random Forest'].predict(df_manual_encoded)[0]
            pred_dt = models['Decision Tree'].predict(df_manual_encoded)[0]
            
            pred_rf_proba = models['Random Forest'].predict_proba(df_manual_encoded)[0]
            pred_dt_proba = models['Decision Tree'].predict_proba(df_manual_encoded)[0]
            
            label_rf = target_le.inverse_transform([pred_rf])[0]
            label_dt = target_le.inverse_transform([pred_dt])[0]
            
            # Tampilkan hasil
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.success(f"🌲 **Random Forest:** {label_rf}")
                
                # Probabilitas RF
                prob_df_rf = pd.DataFrame({
                    'Kelas': labels,
                    'Probabilitas': pred_rf_proba
                }).sort_values('Probabilitas', ascending=False)
                st.dataframe(prob_df_rf, use_container_width=True, hide_index=True)
            
            with col_result2:
                st.success(f"🌳 **Decision Tree:** {label_dt}")
                
                # Probabilitas DT
                prob_df_dt = pd.DataFrame({
                    'Kelas': labels,
                    'Probabilitas': pred_dt_proba
                }).sort_values('Probabilitas', ascending=False)
                st.dataframe(prob_df_dt, use_container_width=True, hide_index=True)
            
            # SHAP explanation untuk Random Forest
            st.markdown("---")
            st.markdown("**🧠 Interpretasi SHAP (Random Forest):**")
            
            try:
                fig_shap, _, _ = generate_shap_plot(
                    models['Random Forest'], 
                    df_manual_encoded, 
                    0, 
                    pred_rf, 
                    [FEATURE_LABELS[f] for f in FEATURE_COLUMNS]
                )
                st.pyplot(fig_shap)
                plt.close()
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
                
        except Exception as e:
            st.error(f"❌ Error dalam prediksi: {e}")

# ==========================================
# EVALUASI MODEL
# ==========================================

st.markdown("---")
st.subheader("📊 Evaluasi Model")

if 'label_efektivitas' in df.columns:
    # Persiapan data evaluasi
    y_true = target_le.transform(df['label_efektivitas'])
    y_pred_rf = target_le.transform(df['Prediksi_RF'])
    y_pred_dt = target_le.transform(df['Prediksi_DT'])
    
    # Tabs untuk evaluasi
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["🌲 Random Forest", "🌳 Decision Tree", "⚖️ Perbandingan"])
    
    with eval_tab1:
        st.markdown("### Random Forest")
        
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            # Akurasi
            accuracy_rf = (y_pred_rf == y_true).mean()
            st.metric("Akurasi", f"{accuracy_rf:.3f}")
            
            # Confusion Matrix
            cm_rf = confusion_matrix(y_true, y_pred_rf)
            fig_cm_rf, ax_cm_rf = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=labels).plot(ax=ax_cm_rf)
            ax_cm_rf.set_title("Confusion Matrix - Random Forest")
            st.pyplot(fig_cm_rf)
            plt.close()
        
        with col_metric2:
            # Classification Report
            report_rf = classification_report(
                y_true, y_pred_rf,
                target_names=labels,
                output_dict=True,
                zero_division=0
            )
            
            st.markdown("**Classification Report:**")
            report_df_rf = pd.DataFrame(report_rf).transpose().round(3)
            st.dataframe(report_df_rf, use_container_width=True)
            
            # Feature Importance
            st.markdown("**Feature Importance:**")
            feat_imp_rf = pd.DataFrame({
                'Feature': [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
                'Importance': models['Random Forest'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp_rf, ax_imp_rf = plt.subplots(figsize=(8, 5))
            feat_imp_rf.plot(x='Feature', y='Importance', kind='barh', ax=ax_imp_rf)
            ax_imp_rf.set_title('Feature Importance - Random Forest')
            plt.tight_layout()
            st.pyplot(fig_imp_rf)
            plt.close()
    
    with eval_tab2:
        st.markdown("### Decision Tree")
        
        col_metric3, col_metric4 = st.columns(2)
        
        with col_metric3:
            # Akurasi
            accuracy_dt = (y_pred_dt == y_true).mean()
            st.metric("Akurasi", f"{accuracy_dt:.3f}")
            
            # Confusion Matrix
            cm_dt = confusion_matrix(y_true, y_pred_dt)
            fig_cm_dt, ax_cm_dt = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=labels).plot(ax=ax_cm_dt)
            ax_cm_dt.set_title("Confusion Matrix - Decision Tree")
            st.pyplot(fig_cm_dt)
            plt.close()
        
        with col_metric4:
            # Classification Report
            report_dt = classification_report(
                y_true, y_pred_dt,
                target_names=labels,
                output_dict=True,
                zero_division=0
            )
            
            st.markdown("**Classification Report:**")
            report_df_dt = pd.DataFrame(report_dt).transpose().round(3)
            st.dataframe(report_df_dt, use_container_width=True)
            
            # Feature Importance
            st.markdown("**Feature Importance:**")
            feat_imp_dt = pd.DataFrame({
                'Feature': [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
                'Importance': models['Decision Tree'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp_dt, ax_imp_dt = plt.subplots(figsize=(8, 5))
            feat_imp_dt.plot(x='Feature', y='Importance', kind='barh', ax=ax_imp_dt)
            ax_imp_dt.set_title('Feature Importance - Decision Tree')
            plt.tight_layout()
            st.pyplot(fig_imp_dt)
            plt.close()
    
    with eval_tab3:
        st.markdown("### Perbandingan Model")
        
        # Perbandingan akurasi
        comparison_data = {
            'Model': ['Random Forest', 'Decision Tree'],
            'Akurasi': [accuracy_rf, accuracy_dt]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Perbandingan F1-Score
        f1_rf = pd.DataFrame(report_rf).transpose()['f1-score'].drop(['accuracy', 'macro avg', 'weighted avg'])
        f1_dt = pd.DataFrame(report_dt).transpose()['f1-score'].drop(['accuracy', 'macro avg', 'weighted avg'])
        
        f1_comparison = pd.DataFrame({
            'Random Forest': f1_rf,
            'Decision Tree': f1_dt
        })
        
        st.markdown("**F1-Score per Kelas:**")
        st.dataframe(f1_comparison.round(3), use_container_width=True)
        
        # Visualisasi perbandingan
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        f1_comparison.plot(kind='bar', ax=ax_comp)
        ax_comp.set_ylabel('F1-Score')
        ax_comp.set_title('Perbandingan F1-Score: Random Forest vs Decision Tree')
        ax_comp.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_comp)
        plt.close()
        
        # Download evaluasi
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Download Excel
            if st.button("📊 Download Evaluasi Excel"):
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    report_df_rf.to_excel(writer, sheet_name="Random_Forest")
                    report_df_dt.to_excel(writer, sheet_name="Decision_Tree") 
                    f1_comparison.to_excel(writer, sheet_name="Perbandingan_F1")
                    comparison_df.to_excel(writer, sheet_name="Akurasi", index=False)
                
                towrite.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=towrite.getvalue(),
                    file_name=f"evaluasi_model_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col_dl2:
            # Info model terbaik
            if accuracy_rf > accuracy_dt:
                st.success("🏆 **Model Terbaik:** Random Forest")
                st.info(f"Akurasi lebih tinggi: {accuracy_rf:.3f} vs {accuracy_dt:.3f}")
            elif accuracy_dt > accuracy_rf:
                st.success("🏆 **Model Terbaik:** Decision Tree")
                st.info(f"Akurasi lebih tinggi: {accuracy_dt:.3f} vs {accuracy_rf:.3f}")
            else:
                st.info("🤝 **Kedua model memiliki akurasi yang sama**")

else:
    st.warning("⚠️ Kolom 'label_efektivitas' tidak ditemukan. Evaluasi tidak dapat dilakukan.")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Skripsi Informatika S1 - UPN "Veteran" Yogyakarta</strong></p>
    <p>Muhammad Rizqi (123190083) | 2025</p>
    <p><em>Klasifikasi Efektivitas Respon Intervensi Stunting di Desa</em></p>
</div>
""", unsafe_allow_html=True)