import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- Load dataset
file_path = "data/stunting_2023_labeled.csv"
df = pd.read_csv(file_path)

fitur_kategori = [
    'Desa_melakukan_monitoring/evaluasi_atas_pelaksanaan_konvergensi_stunting_min._2_kali_dlm_1tahun',
    'Aktivitas_rutin_Penyelenggaraan_posyandu_',
    'Terdapat_Pembentukan_RDS/TPPS',
    'Terdapat_Pelaku_Desa_(Kader,_KPM,_TPK)_mendapatkan_peningkatan_kapasitas',
    'Terdapat_Pengembangan_Program_Ketahanan_Pangan'
]

X = df[fitur_kategori].fillna("Tidak Ada")
X = X.apply(lambda col: col.str.strip().str.lower())

# Encode fitur kategorikal
le_dict = {}
for col in fitur_kategori:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode label target
label_le = LabelEncoder()
y = label_le.fit_transform(df['label_efektivitas'])

# Tampilkan distribusi label sebelum split
print("Distribusi label sebelum split:")
for i, label in enumerate(label_le.classes_):
    print(f"{label}: {(y == i).sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tampilkan distribusi label pada data train sebelum oversampling
print("\nDistribusi label pada data train sebelum oversampling:")
for i, label in enumerate(label_le.classes_):
    print(f"{label}: {(y_train == i).sum()}")

# Oversample data train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Tampilkan distribusi label pada data train setelah oversampling
print("\nDistribusi label pada data train setelah SMOTE:")
for i, label in enumerate(label_le.classes_):
    print(f"{label}: {(y_train_bal == i).sum()}")

# --- Train Decision Tree (pakai data hasil oversampling)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_bal, y_train_bal)
joblib.dump(dt_model, "model/decision_tree_model.pkl")

# --- Train Random Forest (pakai data hasil oversampling)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_bal, y_train_bal)
joblib.dump(rf_model, "model/random_forest_model.pkl")

# --- Simpan LabelEncoder untuk fitur dan target
joblib.dump(le_dict, "model/feature_label_encoders.pkl")
joblib.dump(label_le, "model/target_label_encoder.pkl")
