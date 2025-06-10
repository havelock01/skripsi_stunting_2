import sys
import importlib

# Daftar modul yang wajib tersedia
required_modules = [
    "pandas", "numpy", "sklearn", "matplotlib", "seaborn",
    "joblib", "shap", "streamlit", "plotly", "fpdf"
]

print("🔎 Environment Checker for Skripsi Stunting")
print("=" * 50)
print(f"📦 Python version: {sys.version}")
print("=" * 50)

# Cek setiap modul
for mod in required_modules:
    try:
        imported = importlib.import_module(mod)
        version = getattr(imported, '__version__', 'built-in')
        print(f"✅ {mod.ljust(15)} OK - version {version}")
    except ImportError:
        print(f"❌ {mod.ljust(15)} MISSING - install with: pip install {mod}")

print("=" * 50)
print("📌 Jika ada modul MISSING, jalankan: pip install <nama_modul>")