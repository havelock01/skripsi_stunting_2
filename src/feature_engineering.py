def add_effectiveness_label(df):
    """
    Kolom label efektivitas dengan tiga kategori:
    - Efektif: skor >= 3 dan terdapat anggaran > 0
    - Cukup Efektif: skor 1-2 dan anggaran >= median anggaran non-zero
    - Kurang Efektif: skor 0 atau anggaran < median anggaran non-zero
    """
    indikator_kunci = [
        'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
        'Aktivitas_rutin_Penyelenggaraan_posyandu_',
        'Terdapat_Pembentukan_RDS/TPPS',
        'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
        'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
    ]

    # Isi nilai kosong indikator dengan 'Tidak Ada'
    df[indikator_kunci] = df[indikator_kunci].fillna("Tidak Ada")
    
    # Hitung skor: jumlah indikator dengan nilai 'Ada' atau 'Rutin Tiap Bulan'
    df['skor'] = df[indikator_kunci].apply(lambda row: sum(row == "Ada") + sum(row == "Rutin Tiap Bulan"), axis=1)

    # Logika pelabelan: >= 3 indikator aktif + ada anggaran
    df['label_efektivitas'] = df.apply(
        lambda row: "Efektif"
        if row['skor'] >= 3 and row.get('Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting', 0) > 0
        else "Tidak Efektif",
        axis=1
    )
    
    # Menentukan threshold anggaran: median dari nilai anggaran non-zero
    col_anggaran = 'Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting'
    non_zero = df[df[col_anggaran] > 0][col_anggaran]
    some_minimum = non_zero.quantile(0.50) if len(non_zero) > 0 else 0

    # Fungsi kategorisasi berdasarkan skor dan threshold anggaran
    def categorize(row):
        anggaran = row.get(col_anggaran, 0)
        if row['skor'] >= 3 and anggaran > 0:
            return "Efektif"
        elif row['skor'] >= 1 and anggaran >= some_minimum:
            return "Cukup Efektif"
        else:
            return "Kurang Efektif"

    # Terapkan kategori
    df['label_efektivitas'] = df.apply(categorize, axis=1)

    return df
