from sklearn.preprocessing import LabelEncoder

def fit_label_encoders(df, fitur_kategori):
    le_dict = {}
    df_encoded = df.copy()
    for col in fitur_kategori:
        le = LabelEncoder()
        le.fit(df[col])
        df_encoded[col] = le.transform(df[col])
        le_dict[col] = le
    return df_encoded, le_dict