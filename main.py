from src.preprocessing import load_and_clean_excel
from src.feature_engineering import add_effectiveness_label
import pandas as pd

if __name__ == "__main__":
    path = "data/jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx"
    df = load_and_clean_excel(path)

    # Tambahkan kolom lokasi dari kolom Unnamed (kolom ke-4 hingga ke-6 biasanya)
    df['NAMA_KABUPATEN'] = df.iloc[:, 3]
    df['NAMA_KECAMATAN'] = df.iloc[:, 5]
    df['KODE_DESA'] = df.iloc[:, 6]
    df['NAMA_DESA'] = df.iloc[:, 7]

    df = add_effectiveness_label(df)
    df.to_csv("data/stunting_2023_labeled.csv", index=False)
    print("✅ Data berhasil disimpan dengan kolom lokasi.")
