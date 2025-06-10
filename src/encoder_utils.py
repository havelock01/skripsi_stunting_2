from sklearn.preprocessing import LabelEncoder

def normalize_kategorikal(df, fitur_kategori):
    df = df.copy()
    for col in fitur_kategori:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "tidak ada": "Tidak Ada",
                "tidak": "Tidak Ada",
                "tdk": "Tidak Ada",
                "0": "Tidak Ada",
                "ada": "Ada",
                "iya": "Ada",
                "ya": "Ada",
                "rutin tiap bulan": "Rutin Tiap Bulan",
                "rutin setiap bulan": "Rutin Tiap Bulan"
            })
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
