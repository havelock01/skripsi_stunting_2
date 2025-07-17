from sklearn.preprocessing import LabelEncoder

def normalize_kategorikal(df, fitur_kategori):
    df = df.copy()
    mapping = {
        "tidak ada": "Tidak Ada",
        "tidak": "Tidak Ada",
        "tdk": "Tidak Ada",
        "0": "Tidak Ada",
        "kosong": "Tidak Ada",
        "ada": "Ada",
        "iya": "Ada",
        "ya": "Ada",
        "1": "Ada",
        "rutin tiap bulan": "Rutin Tiap Bulan",
        "rutin setiap bulan": "Rutin Tiap Bulan",
        "rutin": "Rutin Tiap Bulan"
    }
    for col in fitur_kategori:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(mapping)
        )
    return df

def fit_label_encoders(df, fitur_kategori):
    df = normalize_kategorikal(df, fitur_kategori)
    le_dict = {}
    df_encoded = df.copy()
    for col in fitur_kategori:
        le = LabelEncoder()
        le.fit(df_encoded[col])
        df_encoded[col] = le.transform(df_encoded[col])
        le_dict[col] = le
    return df_encoded, le_dict

def encode_features(df, le_dict, features):
    df_enc = normalize_kategorikal(df, features)
    for col in features:
        # Mapping nilai yang tidak ada di encoder ke kelas pertama
        df_enc[col] = df_enc[col].where(df_enc[col].isin(le_dict[col].classes_), le_dict[col].classes_[0])
        df_enc[col] = le_dict[col].transform(df_enc[col])
    return df_enc
