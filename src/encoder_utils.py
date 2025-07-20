"""
Utility functions untuk encoding fitur kategorikal
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def normalize_kategorikal(df, fitur_kategori):
    """
    Normalisasi nilai kategorikal untuk konsistensi
    
    Args:
        df (pd.DataFrame): DataFrame yang akan dinormalisasi
        fitur_kategori (list): List nama kolom kategorikal
    
    Returns:
        pd.DataFrame: DataFrame dengan nilai yang sudah dinormalisasi
    """
    df = df.copy()
    
    # Mapping nilai untuk normalisasi
    mapping = {
        # Nilai negatif
        "tidak ada": "Tidak Ada",
        "tidak": "Tidak Ada",
        "tdk": "Tidak Ada",
        "0": "Tidak Ada",
        "kosong": "Tidak Ada",
        "nan": "Tidak Ada",
        "none": "Tidak Ada",
        
        # Nilai positif
        "ada": "Ada",
        "iya": "Ada",
        "ya": "Ada",
        "1": "Ada",
        "tersedia": "Ada",
        "terdapat": "Ada",
        
        # Nilai rutin
        "rutin tiap bulan": "Rutin Tiap Bulan",
        "rutin setiap bulan": "Rutin Tiap Bulan",
        "rutin": "Rutin Tiap Bulan",
        "bulanan": "Rutin Tiap Bulan"
    }
    
    for col in fitur_kategori:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace(mapping)
            )
    
    return df

def fit_label_encoders(df, fitur_kategori):
    """
    Fit LabelEncoder untuk setiap fitur kategorikal
    
    Args:
        df (pd.DataFrame): DataFrame untuk training encoder
        fitur_kategori (list): List nama kolom kategorikal
    
    Returns:
        tuple: (df_encoded, le_dict)
            - df_encoded: DataFrame yang sudah di-encode
            - le_dict: Dictionary berisi LabelEncoder untuk setiap kolom
    """
    # Normalisasi dulu
    df = normalize_kategorikal(df, fitur_kategori)
    
    le_dict = {}
    df_encoded = df.copy()
    
    for col in fitur_kategori:
        if col in df.columns:
            le = LabelEncoder()
            
            # Fill NaN values dengan "Tidak Ada"
            df_encoded[col] = df_encoded[col].fillna("Tidak Ada")
            
            # Fit dan transform
            le.fit(df_encoded[col])
            df_encoded[col] = le.transform(df_encoded[col])
            
            # Simpan encoder
            le_dict[col] = le
            
            print(f"   ✅ {col}: {len(le.classes_)} kelas unik")
    
    return df_encoded, le_dict

def encode_features(df, le_dict, features):
    """
    Encode fitur menggunakan LabelEncoder yang sudah di-fit
    
    Args:
        df (pd.DataFrame): DataFrame yang akan di-encode
        le_dict (dict): Dictionary berisi LabelEncoder
        features (list): List nama fitur yang akan di-encode
    
    Returns:
        pd.DataFrame: DataFrame yang sudah di-encode
    """
    df_enc = normalize_kategorikal(df[features].copy(), features)
    
    for col in features:
        if col in df_enc.columns and col in le_dict:
            # Fill NaN dengan "Tidak Ada"
            df_enc[col] = df_enc[col].fillna("Tidak Ada")
            
            # Handle nilai yang tidak ada di encoder dengan mapping ke kelas pertama
            valid_classes = le_dict[col].classes_
            df_enc[col] = df_enc[col].where(
                df_enc[col].isin(valid_classes), 
                valid_classes[0]
            )
            
            # Transform
            df_enc[col] = le_dict[col].transform(df_enc[col])
    
    return df_enc

def encode_manual_input(input_dict, le_dict, fitur_kategori):
    """
    Encode input manual dari form
    
    Args:
        input_dict (dict): Dictionary input dari form
        le_dict (dict): Dictionary berisi LabelEncoder
        fitur_kategori (list): List nama fitur kategorikal
    
    Returns:
        pd.DataFrame: DataFrame yang sudah di-encode
    """
    df_manual = pd.DataFrame([input_dict])
    return encode_features(df_manual, le_dict, fitur_kategori)

def get_feature_mapping(le_dict):
    """
    Mendapatkan mapping encoding untuk semua fitur
    
    Args:
        le_dict (dict): Dictionary berisi LabelEncoder
    
    Returns:
        dict: Dictionary berisi mapping untuk setiap fitur
    """
    mapping_dict = {}
    
    for col, le in le_dict.items():
        mapping_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    return mapping_dict

def validate_encoder_consistency(df, le_dict, features):
    """
    Validasi konsistensi encoder dengan data
    
    Args:
        df (pd.DataFrame): DataFrame untuk validasi
        le_dict (dict): Dictionary berisi LabelEncoder
        features (list): List nama fitur
    
    Returns:
        dict: Dictionary berisi hasil validasi
    """
    validation_results = {}
    
    for col in features:
        if col in df.columns and col in le_dict:
            unique_values = set(df[col].dropna().astype(str).str.strip().str.lower().unique())
            encoder_classes = set(le_dict[col].classes_)
            
            missing_in_encoder = unique_values - encoder_classes
            
            validation_results[col] = {
                'unique_values_count': len(unique_values),
                'encoder_classes_count': len(encoder_classes),
                'missing_in_encoder': list(missing_in_encoder),
                'is_consistent': len(missing_in_encoder) == 0
            }
    
    return validation_results