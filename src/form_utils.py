import pandas as pd

def encode_manual_input(input_dict, le_dict, fitur_kategori):
    df_manual = pd.DataFrame([input_dict])
    df_encoded = df_manual.copy()
    for col in fitur_kategori:
        df_encoded[col] = le_dict[col].transform(df_manual[col])
    return df_encoded