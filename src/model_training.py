import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
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
labels_list = list(label_le.classes_) if hasattr(label_le, 'classes_') and label_le.classes_ is not None else []
print("Distribusi label sebelum split:")
for i, label in enumerate(labels_list):
    print(f"{label}: {np.sum(y == i)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tampilkan distribusi label pada data train sebelum oversampling
print("\nDistribusi label pada data train sebelum oversampling:")
for i, label in enumerate(labels_list):
    print(f"{label}: {np.sum(y_train == i)}")

# Oversample data train
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)  # type: ignore

# Tampilkan distribusi label pada data train setelah SMOTE:
print("\nDistribusi label pada data train setelah SMOTE:")
for i, label in enumerate(labels_list):
    print(f"{label}: {np.sum(y_train_bal == i)}")

# Oversample lagi dengan RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_bal, y_train_bal)  # type: ignore
if not isinstance(y_train_ros, np.ndarray):
    y_train_ros = np.array(y_train_ros)
print("\nDistribusi label pada data train setelah RandomOverSampler:")
for i, label in enumerate(labels_list):
    print(f"{label}: {np.sum(y_train_ros == i)}")

# Debug: print shape dan cek NaN
print("X_train_bal shape:", X_train_bal.shape)
print("X_train_ros shape:", X_train_ros.shape)
print("y_train_bal shape:", y_train_bal.shape)
print("y_train_ros shape:", y_train_ros.shape)
print("NaN in X_train_bal:", np.isnan(X_train_bal).any() if hasattr(X_train_bal, 'any') else False)
print("NaN in X_train_ros:", np.isnan(X_train_ros).any() if hasattr(X_train_ros, 'any') else False)
print("NaN in y_train_bal:", np.isnan(y_train_bal).any() if hasattr(y_train_bal, 'any') else False)
print("NaN in y_train_ros:", np.isnan(y_train_ros).any() if hasattr(y_train_ros, 'any') else False)

# --- Train Balanced Random Forest (pakai data hasil RandomOverSampler)
bal_rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
bal_rf_model.fit(X_train_ros, y_train_ros)
joblib.dump(bal_rf_model, "model/balanced_random_forest_model.pkl")

# --- Train Decision Tree (pakai data hasil oversampling)
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train_bal, y_train_bal)
joblib.dump(dt_model, "model/decision_tree_model.pkl")

# --- Train Random Forest (pakai data hasil oversampling)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_bal, y_train_bal)
joblib.dump(rf_model, "model/random_forest_model.pkl")

# --- Simpan LabelEncoder untuk fitur dan target
joblib.dump(le_dict, "model/feature_label_encoders.pkl")
joblib.dump(label_le, "model/target_label_encoder.pkl")
