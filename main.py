import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing import load_and_clean_excel
from src.feature_engineering import add_effectiveness_label
import pandas as pd

def main():
    """Main function untuk preprocessing data stunting"""
    
    print("🎓 Skripsi: Klasifikasi Efektivitas Intervensi Stunting")
    print("=" * 60)
    print("📊 Memulai preprocessing data...")
    
    # Path file data
    data_path = "data/jumlah-penerima-layanan-pencegahan-stunting-tahun-2023.xlsx"
    output_path = "data/stunting_2023_labeled.csv"
    
    # Check if input file exists
    if not os.path.exists(data_path):
        print(f"❌ File data tidak ditemukan: {data_path}")
        print("   Pastikan file Excel sudah ada di folder data/")
        return False
    
    try:
        # 1. Load dan clean Excel data
        print("🔄 Loading dan cleaning data Excel...")
        df = load_and_clean_excel(data_path)
        print(f"   ✅ Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
        
        # 2. Tambahkan kolom lokasi dari kolom Unnamed
        print("🔄 Menambahkan kolom lokasi...")
        df['NAMA_KABUPATEN'] = df.iloc[:, 3]
        df['NAMA_KECAMATAN'] = df.iloc[:, 5] 
        df['KODE_DESA'] = df.iloc[:, 6]
        df['NAMA_DESA'] = df.iloc[:, 7]
        
        # Cek apakah kolom lokasi berhasil ditambahkan
        lokasi_cols = ['NAMA_KABUPATEN', 'NAMA_KECAMATAN', 'KODE_DESA', 'NAMA_DESA']
        for col in lokasi_cols:
            non_null_count = df[col].notna().sum()
            print(f"   - {col}: {non_null_count} entries")
        
        # 3. Feature engineering: tambahkan label efektivitas
        print("🔄 Menambahkan label efektivitas...")
        df = add_effectiveness_label(df)
        
        # 4. Simpan hasil ke CSV
        print(f"💾 Menyimpan hasil ke: {output_path}")
        
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Simpan ke CSV
        df.to_csv(output_path, index=False)
        
        # 5. Tampilkan ringkasan hasil
        print("\n📊 Ringkasan Hasil Preprocessing:")
        print(f"   - Total desa: {len(df)}")
        print(f"   - Kolom total: {len(df.columns)}")
        print(f"   - File output: {output_path}")
        
        # Distribusi label efektivitas
        if 'label_efektivitas' in df.columns:
            print("\n📈 Distribusi Label Efektivitas:")
            label_dist = df['label_efektivitas'].value_counts()
            for label, count in label_dist.items():
                percentage = (count / len(df)) * 100
                print(f"   - {label}: {count} ({percentage:.1f}%)")
        
        # Distribusi skor
        if 'skor' in df.columns:
            print("\n📈 Distribusi Skor Efektivitas:")
            skor_dist = df['skor'].value_counts().sort_index()
            for skor, count in skor_dist.items():
                percentage = (count / len(df)) * 100
                print(f"   - Skor {skor}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Preprocessing selesai! Data siap untuk training model.")
        print(f"   Jalankan: python src/model_training.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)