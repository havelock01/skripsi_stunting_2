def add_effectiveness_label(df):
    indikator_kunci = [
        'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
        'Aktivitas_rutin_Penyelenggaraan_posyandu_',
        'Terdapat_Pembentukan_RDS/TPPS',
        'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
        'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
    ]

    df[indikator_kunci] = df[indikator_kunci].fillna("Tidak Ada")
    df['skor'] = df[indikator_kunci].apply(lambda row: sum(row == "Ada") + sum(row == "Rutin Tiap Bulan"), axis=1)

    # Logika pelabelan: >= 3 indikator aktif + ada anggaran
    df['label_efektivitas'] = df.apply(
        lambda row: "Efektif"
        if row['skor'] >= 3 and row.get('Jumlah_alokasi_anggaran_untuk_mendukung_kegiatan_stunting', 0) > 0
        else "Tidak Efektif",
        axis=1
    )

    return df
