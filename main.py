from src.preprocessing import load_and_clean_excel
from src.feature_engineering import add_effectiveness_label
import pandas as pd

if __name__ == "__main__":
    path = "data/jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx"
    df = load_and_clean_excel(path)
    df = add_effectiveness_label(df)
    df.to_csv("data/stunting_2023_labeled.csv", index=False)
    print("✅ Data berhasil disiapkan dan disimpan.")