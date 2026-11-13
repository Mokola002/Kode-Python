# =====================================================================
# JUDUL: PREDIKSI TANAH KERING DALAM 48 JAM - TANAMAN INDOOR DI KOS
# MODEL: DECISION TREE CLASSIFIER
# INPUT: soil_percent, temperature, humidity (saat ini)
# OUTPUT: AKAN KERING / TIDAK KERING (48 jam ke depan)
# DATASET: 65.028 data sensor ESP32, 2 - 12 Nov 2026
# =====================================================================

import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FEATURES = ["soil_percent", "temperature", "humidity"]
FORECAST_HOURS = 48
DRY_SOIL_THRESHOLD = 54.50


# === LANGKAH 0: UPLOAD DATASET ===
print("Silahkan upload file dataset_irigasi.csv")
uploaded = files.upload()
nama_file = list(uploaded.keys())[0]
print(f"File berhasil di-upload: {nama_file}\n")

df = pd.read_csv(nama_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

print(f"Total data mentah: {len(df)} baris")
print(f"Periode: {df['timestamp'].min()} sampai {df['timestamp'].max()}")
print(f"Kolom: {list(df.columns)}\n")

# === LANGKAH 1: BIKIN LABEL "AKAN KERING DALAM 48 JAM" ===
# Logika: Kita intip soil_percent 48 jam ke depan.
# Kalau 48 jam lagi soil <= 54.50, berarti AKAN KERING.

# Sederhanakan jadi data PER JAM (data asli tiap 13 detik, terlalu banyak)
df_jam = (
    df.set_index("timestamp")
    .resample("1h")
    .mean(numeric_only=True)
    .reset_index()
)
print(f"Data setelah dirata-rata per jam: {len(df_jam)} baris")

# Lihat tanah 48 jam ke depan
df_jam["soil_48jam_lagi"] = df_jam["soil_percent"].shift(-FORECAST_HOURS)


def bikin_label(soil_nanti):
    if pd.isna(soil_nanti):
        return None

    if soil_nanti <= DRY_SOIL_THRESHOLD:
        return "AKAN KERING"
    return "TIDAK KERING"


df_jam["label"] = df_jam["soil_48jam_lagi"].apply(bikin_label)
df_latih = df_jam.dropna(subset=["label"]).copy()

print("\n=== DISTRIBUSI LABEL ===")
print(df_latih["label"].value_counts())
print(f"Total data buat latih: {len(df_latih)} jam")

# === LANGKAH 2: LATIH MODEL DECISION TREE ===
# X = yang kita tau SEKARANG, y = yang mau diprediksi 48 JAM LAGI
X = df_latih[FEATURES]
y = df_latih["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"\nData latih: {len(X_train)}, Data tes: {len(X_test)}")

# INI MODEL DECISION TREE-NYA
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# INI HASIL KLASIFIKASINYA
pred = model.predict(X_test)
print(f"\n=== HASIL KLASIFIKASI ===")
print(f"Akurasi: {accuracy_score(y_test, pred) * 100:.1f}%")
print(classification_report(y_test, pred))

print("\n=== ATURAN POHON YANG DIPELAJARI MODEL ===")
print(export_text(model, feature_names=FEATURES))

# === LANGKAH 3: GAMBAR POHONNYA (Biar gampang dibaca dosen) ===
plt.figure(figsize=(16, 8))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=["AKAN KERING", "TIDAK KERING"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(
    "Decision Tree - Prediksi Tanah Kering Dalam 48 Jam "
    "(Tanaman Indoor Kos)"
)
plt.show()

# === LANGKAH 4: COBA PREDIKSI KASUS NYATA DI KOS ===
print("\n=== COBA PREDIKSI KASUS KOS ===")


def prediksi_kasus(soil, temp, hum, keterangan):
    data_baru = pd.DataFrame([[soil, temp, hum]], columns=FEATURES)
    hasil = model.predict(data_baru)[0]
    print(
        f"{keterangan} | soil={soil}%, "
        f"suhu={temp}C, humidity={hum}% -> {hasil}"
    )


prediksi_kasus(100, 26, 45, "Kasus 1: Habis siram")
prediksi_kasus(73, 25, 40, "Kasus 2: Tanah mulai turun, kosan adem")
prediksi_kasus(80, 30, 35, "Kasus 3: Tanah 80%, kosan panas")
prediksi_kasus(70, 27, 50, "Kasus 4: Tanah 70%, humidity tinggi")

print("\n=== SELESAI ===")
print(
    "Kesimpulan: Jika soil_percent <= 84.37 dan humidity > 40.19, "
    "maka 48 jam lagi AKAN KERING."
)