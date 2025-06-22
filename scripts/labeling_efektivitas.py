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
df['label_efektivitas'] = df.apply(
        lambda row: "Efektif"
        if row['skor'] >= 3 and row.get('Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting', 0) > 0
        else "Tidak Efektif",
        axis=1
    )

# 5. Fungsi untuk memberi kategori berdasarkan skor dan anggaran
def categorize(row):
    anggaran = row.get('Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting', 0)
    if anggaran > 0:
        if row['skor'] >= 3:
            return "Efektif"
        elif row['skor'] >= 1:
            return "Cukup Efektif"
        else:
            return "Kurang Efektif"
    else:
        return "Kurang Efektif"

# Terapkan kategori
df['label_efektivitas'] = df.apply(categorize, axis=1)

# 6. Simpan hasil ke file baru
output_file = "data/stunting_2023_labeled.csv"
df.to_csv(output_file, index=False)

# 7. Tampilkan hasil distribusi label
print("Distribusi Efektivitas Intervensi:")
print(df['label_efektivitas'].value_counts())
print(f"\n✅ File hasil labeling berhasil disimpan ke: {output_file}")
