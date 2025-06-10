import pandas as pd

# 1. Load file CSV hasil cleaning
file_path = "data/stunting_2023_cleaned.csv"
df = pd.read_csv(file_path)

# Rename kolom lokasi agar lebih mudah digunakan
df.rename(columns={
    'Unnamed: 3': 'NAMA_KABUPATEN',
    'Unnamed: 4': 'KODE_KECAMATAN',
    'Unnamed: 5': 'NAMA_KECAMATAN',
    'Unnamed: 6': 'KODE_DESA',
    'Unnamed: 7': 'NAMA_DESA',
    'Unnamed: 8': 'TAHUN_DATA'
}, inplace=True)


# 2. Daftar indikator utama untuk penilaian efektivitas
indikator_kunci = [
    'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
    'Aktivitas_rutin_Penyelenggaraan_posyandu_',
    'Terdapat_Pembentukan_RDS/TPPS',
    'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
    'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
]

# 3. Tangani nilai NaN (kosong) agar tidak menyebabkan error
df[indikator_kunci] = df[indikator_kunci].fillna("Tidak Ada")

# 4. Hitung skor (berapa banyak indikator aktif per desa)
df['skor'] = df[indikator_kunci].apply(
    lambda row: sum(row == "Ada") + sum(row == "Rutin Tiap Bulan"), axis=1
)

# 5. Bentuk label klasifikasi berdasarkan skor
df['label_efektivitas'] = df['skor'].apply(
    lambda x: "Efektif" if x >= 4 else "Tidak Efektif"
)

# 6. Simpan hasil ke file baru
output_file = "data/stunting_2023_labeled.csv"
df.to_csv(output_file, index=False)

# 7. Tampilkan hasil distribusi label
print("Distribusi Efektivitas Intervensi:")
print(df['label_efektivitas'].value_counts())
print(f"\n✅ File hasil labeling berhasil disimpan ke: {output_file}")
