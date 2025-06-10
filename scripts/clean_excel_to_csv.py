import pandas as pd

# 1. Baca file Excel
file_path = "data/jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Sheet1")

# 2. Buang baris header ganda (baris 0 dan 1)
df_clean = df_raw.drop(index=[0, 1]).reset_index(drop=True)

# 3. Set ulang nama kolom dari baris ke-2 (index 1 sebelumnya)
df_clean.columns = df_raw.iloc[1]
df_clean = df_clean.drop(index=0).reset_index(drop=True)

# print("Kolom tersedia:\n", df_clean.columns.tolist())

# 4. Konversi kolom numerik (contoh beberapa kolom layanan)
numeric_columns = [col for col in df_clean.columns if "Total" in str(col)]
df_clean[numeric_columns] = df_clean[numeric_columns].apply(pd.to_numeric, errors='coerce')

# 5. Simpan ke CSV
output_csv = "data/stunting_2023_cleaned.csv"
df_clean.to_csv(output_csv, index=False)

print(f"Berhasil disimpan ke: {output_csv}")
