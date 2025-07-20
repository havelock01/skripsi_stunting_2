#!/usr/bin/env python3
"""
Script untuk training model Decision Tree dan Random Forest
Skripsi: Klasifikasi Efektivitas Intervensi Stunting di Desa
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

from encoder_utils import normalize_kategorikal

def create_model_directory():
    """Membuat direktori model jika belum ada"""
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Direktori '{model_dir}' telah dibuat.")
    return model_dir

def load_and_prepare_data():
    """Load dan prepare data untuk training"""
    
    # Check file existence
    file_path = "data/stunting_2023_labeled.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} tidak ditemukan! Jalankan main.py terlebih dahulu.")
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"   Dataset shape: {df.shape}")
    
    # Fitur yang akan digunakan
    fitur_kategori = [
        'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
        'Aktivitas_rutin_Penyelenggaraan_posyandu_',
        'Terdapat_Pembentukan_RDS/TPPS',
        'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
        'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
    ]
    
    # Check if all features exist
    missing_features = [f for f in fitur_kategori if f not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Fitur tidak ditemukan: {missing_features}")
    
    # Check if target exists
    if 'label_efektivitas' not in df.columns:
        raise ValueError("❌ Kolom 'label_efektivitas' tidak ditemukan!")
    
    return df, fitur_kategori

def preprocess_features(df, fitur_kategori):
    """Preprocessing fitur untuk training"""
    
    print("🔄 Preprocessing fitur...")
    
    # Extract features
    X = df[fitur_kategori].copy().fillna("Tidak Ada")
    
    # Normalisasi fitur kategorikal
    X = normalize_kategorikal(X, fitur_kategori)
    
    # Encode fitur kategorikal
    le_dict = {}
    X_encoded = X.copy()
    
    print("🔄 Encoding fitur kategorikal...")
    for col in fitur_kategori:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        le_dict[col] = le
        
        # Print mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"   - {col[:50]}...")
        print(f"     Mapping: {mapping}")
    
    return X_encoded, le_dict

def preprocess_target(df):
    """Preprocessing target variable"""
    
    print("🔄 Preprocessing target variable...")
    
    # Encode target
    label_le = LabelEncoder()
    y = label_le.fit_transform(df['label_efektivitas'])
    
    # Print distribusi
    print("📊 Distribusi label:")
    for i, label in enumerate(label_le.classes_):
        count = (y == i).sum()
        percentage = (count / len(y)) * 100
        print(f"   - {label}: {count} ({percentage:.1f}%)")
    
    return y, label_le

def split_and_balance_data(X, y, test_size=0.2, random_state=42):
    """Split data dan handle class imbalance"""
    
    print("🔄 Splitting data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Handle class imbalance dengan SMOTE
    print("🔄 Handling class imbalance dengan SMOTE...")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"   Setelah SMOTE: {len(X_train_balanced)} samples")
    
    # Print distribusi setelah balancing
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print("   Distribusi setelah SMOTE:")
    for label_idx, count in zip(unique, counts):
        print(f"     - Label {label_idx}: {count}")
    
    return X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced

def train_decision_tree(X_train, y_train):
    """Training Decision Tree model"""
    
    print("🌳 Training Decision Tree...")
    
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced'
    )
    
    dt_model.fit(X_train, y_train)
    
    print("   ✅ Decision Tree training selesai")
    return dt_model

def train_random_forest(X_train, y_train):
    """Training Random Forest model"""
    
    print("🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    print("   ✅ Random Forest training selesai")
    return rf_model

def evaluate_models(models, X_test, y_test, label_le):
    """Evaluasi semua model"""
    
    print("\n📊 EVALUASI MODEL")
    print("=" * 50)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Evaluasi {name}:")
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Akurasi
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Akurasi: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=label_le.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Print summary metrics
        print(f"   Precision (macro): {report['macro avg']['precision']:.4f}")
        print(f"   Recall (macro): {report['macro avg']['recall']:.4f}")
        print(f"   F1-Score (macro): {report['macro avg']['f1-score']:.4f}")
        
        # Cross validation
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        print(f"   CV Score (mean±std): {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'cv_scores': cv_scores,
            'predictions': y_pred
        }
    
    return results

def save_models_and_encoders(models, le_dict, label_le, model_dir):
    """Simpan semua model dan encoder"""
    
    print("\n💾 Menyimpan model dan encoder...")
    
    try:
        # Simpan models
        for name, model in models.items():
            filename = f"{model_dir}/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"   ✅ {filename}")
        
        # Simpan encoders
        encoders_file = f"{model_dir}/feature_label_encoders.pkl"
        joblib.dump(le_dict, encoders_file)
        print(f"   ✅ {encoders_file}")
        
        target_encoder_file = f"{model_dir}/target_label_encoder.pkl"
        joblib.dump(label_le, target_encoder_file)
        print(f"   ✅ {target_encoder_file}")
        
        print("✅ Semua model dan encoder berhasil disimpan!")
        
    except Exception as e:
        print(f"❌ Error menyimpan model: {e}")
        raise

def validate_saved_models(model_dir, X_test, label_le):
    """Validasi model yang sudah disimpan"""
    
    print("\n🔍 Validasi model yang disimpan...")
    
    try:
        # Test load models
        dt_loaded = joblib.load(f"{model_dir}/decision_tree_model.pkl")
        rf_loaded = joblib.load(f"{model_dir}/random_forest_model.pkl")
        le_dict_loaded = joblib.load(f"{model_dir}/feature_label_encoders.pkl")
        label_le_loaded = joblib.load(f"{model_dir}/target_label_encoder.pkl")
        
        # Test predictions
        test_pred_dt = dt_loaded.predict(X_test[:5])
        test_pred_rf = rf_loaded.predict(X_test[:5])
        
        test_labels_dt = label_le_loaded.inverse_transform(test_pred_dt)
        test_labels_rf = label_le_loaded.inverse_transform(test_pred_rf)
        
        print("   ✅ Model berhasil di-load")
        print(f"   ✅ Test prediksi DT: {test_labels_dt}")
        print(f"   ✅ Test prediksi RF: {test_labels_rf}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def print_feature_importance(rf_model, fitur_kategori):
    """Print feature importance dari Random Forest"""
    
    print("\n🔍 Feature Importance (Random Forest):")
    print("-" * 50)
    
    importance_df = pd.DataFrame({
        'Feature': fitur_kategori,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        feature_short = row['Feature'][:60] + "..." if len(row['Feature']) > 60 else row['Feature']
        print(f"   {feature_short}: {row['Importance']:.4f}")

def main():
    """Main function untuk training"""
    
    print("🎓 TRAINING MODEL KLASIFIKASI STUNTING")
    print("=" * 60)
    print("📚 Skripsi: Muhammad Rizqi (123190083)")
    print("🏫 Informatika S1, UPN Veteran Yogyakarta")
    print("=" * 60)
    
    try:
        # 1. Setup
        model_dir = create_model_directory()
        
        # 2. Load data
        df, fitur_kategori = load_and_prepare_data()
        
        # 3. Preprocess features
        X_encoded, le_dict = preprocess_features(df, fitur_kategori)
        
        # 4. Preprocess target
        y, label_le = preprocess_target(df)
        
        # 5. Split and balance data
        X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = split_and_balance_data(X_encoded, y)
        
        # 6. Train models
        dt_model = train_decision_tree(X_train_balanced, y_train_balanced)
        rf_model = train_random_forest(X_train_balanced, y_train_balanced)
        
        models = {
            'Decision Tree': dt_model,
            'Random Forest': rf_model
        }
        
        # 7. Evaluate models
        results = evaluate_models(models, X_test, y_test, label_le)
        
        # 8. Print feature importance
        print_feature_importance(rf_model, fitur_kategori)
        
        # 9. Save models
        save_models_and_encoders(models, le_dict, label_le, model_dir)
        
        # 10. Validate saved models
        validation_success = validate_saved_models(model_dir, X_test, label_le)
        
        if validation_success:
            print("\n🎉 TRAINING BERHASIL DISELESAIKAN!")
            print("📱 Jalankan aplikasi Streamlit: streamlit run streamlit_app/app.py")
        else:
            print("\n❌ Validasi model gagal")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error dalam training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)