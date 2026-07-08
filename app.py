# -*- coding: utf-8 -*-
"""
tugas_kelompok6_fixed.py
Versi PERBAIKAN: menggunakan data ASLI dari complete_data.csv
(bukan data simulasi/random seperti versi sebelumnya)
"""

# ==============================================================================
# IMPORT LIBRARY & SETUP ENVIRONMENT
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


# ==============================================================================
# TAHAP 1: PROBLEM DEFINITION
# ==============================================================================
print("="*60)
print("TAHAP 1: PROBLEM DEFINITION")
print("="*60)
print("""
Rumusan Masalah:
1. Bagaimana karakteristik persebaran penduduk usia sekolah, luas wilayah, dan
   jumlah sekolah di Indonesia?
2. Apakah terdapat hubungan linear yang kuat antara jumlah penduduk usia sekolah
   dan luas wilayah terhadap jumlah sekolah?
3. Faktor mana yang memberikan kontribusi paling signifikan terhadap
   pembangunan sekolah baru?

Hipotesis:
H0: Tidak ada pengaruh signifikan antara penduduk usia sekolah dan luas wilayah
    terhadap jumlah sekolah.
H1: Terdapat pengaruh positif dan signifikan dari penduduk usia sekolah dan
    luas wilayah terhadap jumlah sekolah.
""")


# ==============================================================================
# TAHAP 2: DATA COLLECTION  (DIPERBAIKI: pakai data asli, bukan simulasi)
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 2: DATA COLLECTION")
print("="*60)

# --- Baca dataset asli hasil pengumpulan data sekolah se-Indonesia ---
# File berisi data per SEKOLAH (baris = satu sekolah), sehingga perlu
# diagregasi ke level PROVINSI agar sesuai unit analisis (Y = jumlah sekolah
# per provinsi, X = penduduk usia sekolah & luas wilayah per provinsi).
df_raw = pd.read_csv("complete_data.csv")

print("=== INSPEKSI AWAL DATA MENTAH (LEVEL SEKOLAH) ===")
print(f"Dimensi Dataset Mentah: {df_raw.shape[0]} Baris (Sekolah), {df_raw.shape[1]} Kolom")
print(f"Jumlah Provinsi Unik: {df_raw['province_name'].nunique()}")
display(df_raw.head())

# Agregasi ke level provinsi
df = (
    df_raw.groupby("province_name")
    .agg(
        jumlah_sekolah=("school_name", "count"),           # Y  : total sekolah per provinsi
        luas_wilayah=("province_area", "first"),            # X2 : luas wilayah (km2)
        total_penduduk=("total_population", "first"),       # total penduduk (untuk cek VIF)
        penduduk_usia_sekolah=("total_education_age_population", "first"),  # X1
    )
    .reset_index()
    .rename(columns={"province_name": "provinsi"})
)

print("\n=== DATA SETELAH AGREGASI KE LEVEL PROVINSI ===")
print(f"Dimensi Dataset: {df.shape[0]} Baris (Provinsi), {df.shape[1]} Kolom")
display(df.head(10))


# ==============================================================================
# TAHAP 3: DATA CLEANING
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 3: DATA CLEANING")
print("="*60)

print("=== Pengecekan Missing Values (data mentah, kolom lat/long) ===")
print(df_raw.isnull().sum())
print("Catatan: missing value pada lat/long tidak memengaruhi agregasi provinsi\n"
      "karena kolom tersebut tidak dipakai dalam model.")

print("\n=== Pengecekan Missing Values (data agregat provinsi) ===")
print(df.isnull().sum())

print("\n=== Pengecekan Duplikasi Provinsi ===")
print(f"Jumlah baris duplikat: {df.duplicated(subset=['provinsi']).sum()}")

# Validasi logis: Data numerik tidak boleh bernilai negatif
kolom_numerik = ["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah", "jumlah_sekolah"]
for col in kolom_numerik:
    df = df[df[col] >= 0]

print(f"\nDimensi Dataset setelah cleaning: {df.shape[0]} Baris (Provinsi)")


# ==============================================================================
# TAHAP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 4: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("=== TABEL STATISTIK DESKRIPTIF NASIONAL (34 Provinsi) ===")
deskriptif = df[kolom_numerik].describe().T[["mean", "std", "min", "50%", "max"]]
deskriptif.columns = ["Rata-rata", "Std Deviasi", "Minimum", "Median", "Maksimum"]
display(deskriptif)

# Visualisasi Matriks Korelasi
corr_matrix = df[kolom_numerik].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    annot_kws={"size": 12},
)
plt.title("Matriks Korelasi Pearson Antar Variabel", fontsize=14, pad=15, fontweight="bold")
plt.tight_layout()
plt.savefig("corr_matrix.png", dpi=150)
plt.show()

print(f"""
[KESIMPULAN ANALISIS MATRIKS KORELASI]
1. Korelasi total_penduduk vs penduduk_usia_sekolah: {corr_matrix.loc['total_penduduk','penduduk_usia_sekolah']:.2f}
   (rasional karena usia sekolah adalah subset dari total penduduk, berpotensi
   memicu multikolinieritas bila keduanya dimasukkan bersamaan ke regresi).
2. Korelasi penduduk_usia_sekolah vs jumlah_sekolah: {corr_matrix.loc['penduduk_usia_sekolah','jumlah_sekolah']:.2f}
3. Korelasi luas_wilayah vs jumlah_sekolah: {corr_matrix.loc['luas_wilayah','jumlah_sekolah']:.2f}
""")


# ==============================================================================
# TAHAP 5: STATISTICAL MODELING
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 5: STATISTICAL MODELING")
print("="*60)

# Uji VIF
X_check = df[["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah"]]
X_check_const = sm.add_constant(X_check)

vif_table = pd.DataFrame()
vif_table["Variabel Independen"] = X_check_const.columns
vif_table["Nilai VIF"] = [
    variance_inflation_factor(X_check_const.values, i)
    for i in range(X_check_const.shape[1])
]

print("=== UJI ASUMSI MULTIKOLINIERITAS (VIF) ===")
display(vif_table)
print("-> Keputusan Analitik: Variabel total_penduduk di-drop pada pemodelan\n"
      "   karena redundant dengan penduduk_usia_sekolah (VIF tinggi).\n")

# Model Regresi OLS Final
X_final = df[["penduduk_usia_sekolah", "luas_wilayah"]]
X_final = sm.add_constant(X_final)
y = df["jumlah_sekolah"]

model_regresi = sm.OLS(y, X_final).fit()
print("=== RINGKASAN MODEL REGRESI (OLS) ===")
print(model_regresi.summary())

# Plot Residual
df["prediksi"] = model_regresi.predict(X_final)
df["residual"] = y - df["prediksi"]

plt.figure(figsize=(9, 5))
sns.scatterplot(x=df["prediksi"], y=df["residual"], color="red", s=70, alpha=0.7)
plt.axhline(y=0, color="black", linestyle="--")
plt.title("Plot Residual vs Fitted Values (Uji Homoskedastisitas)", fontweight="bold")
plt.xlabel("Nilai Prediksi Jumlah Sekolah")
plt.ylabel("Residual (Error)")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.show()

print("""
[CATATAN INTERPRETASI GRAFIK RESIDUAL]
Periksa langsung dari plot yang dihasilkan (bukan diasumsikan):
- Apakah titik-titik tersebar acak di sekitar garis nol tanpa pola corong?
  Jika ya, asumsi homoskedastisitas relatif terpenuhi.
- Karena n hanya 34 (provinsi), interpretasi visual sebaiknya didukung uji
  formal (mis. Breusch-Pagan) bila diperlukan tingkat keyakinan lebih tinggi.
""")

# Uji formal heteroskedastisitas (tambahan, karena n kecil rawan salah baca visual)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model_regresi.resid, X_final)
labels = ["LM Statistic", "LM p-value", "F-Statistic", "F p-value"]
print("=== UJI BREUSCH-PAGAN (Homoskedastisitas) ===")
for label, value in zip(labels, bp_test):
    print(f"{label}: {value:.4f}")


# ==============================================================================
# TAHAP 6: INTERPRET & COMMUNICATE
# ==============================================================================
print("\n" + "="*60)
print("TAHAP 6: INTERPRET & COMMUNICATE")
print("="*60)

p_x1 = model_regresi.pvalues["penduduk_usia_sekolah"]
p_x2 = model_regresi.pvalues["luas_wilayah"]
r2 = model_regresi.rsquared
f_p = model_regresi.f_pvalue

print(f"""
[INSIGHT & REKOMENDASI FINAL - berdasarkan hasil regresi aktual]

1. Validasi Hipotesis (F-test p-value = {f_p:.4g}):
   {"Model signifikan secara simultan, H0 ditolak." if f_p < 0.05 else "Model TIDAK signifikan secara simultan pada alpha 5%, H0 gagal ditolak."}

2. Signifikansi Parsial:
   - penduduk_usia_sekolah: p-value = {p_x1:.4g} ({"signifikan" if p_x1 < 0.05 else "tidak signifikan"} pada alpha 5%)
   - luas_wilayah: p-value = {p_x2:.4g} ({"signifikan" if p_x2 < 0.05 else "tidak signifikan"} pada alpha 5%)

3. Kekuatan Model:
   R-squared = {r2:.4f}, artinya model menjelaskan {r2*100:.1f}% variasi jumlah sekolah antar provinsi.

4. Keterbatasan & Saran Eksplorasi:
   - Unit analisis hanya n=34 provinsi, sehingga daya statistik (power) terbatas.
   - Penelitian ini belum mengontrol aspek fiskal daerah (mis. APBD) dan
     tipografi geografis (kepulauan vs kontinental).
   - Data 'status' (Negeri/Swasta) dan 'stage' (SD/SMP/SMA) pada data mentah
     bisa dieksplorasi lebih lanjut untuk analisis per jenjang pendidikan.
""")
