# Dashboard Analisis Sekolah Indonesia (Streamlit)

## Isi folder
- `app.py` — aplikasi Streamlit
- `complete_data.csv` — dataset (sudah disertakan, sesuai data yang di-upload)
- `requirements.txt` — daftar library yang dibutuhkan

## Cara menjalankan
1. Pastikan Python sudah terinstal (disarankan Python 3.9+).
2. Buka terminal di folder ini, lalu install library:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```
4. Browser akan otomatis terbuka di `http://localhost:8501`.

## Fitur
- **Overview** — ringkasan nasional, jumlah sekolah per provinsi, komposisi jenjang & status sekolah.
- **EDA & Korelasi** — statistik deskriptif, heatmap korelasi, scatter plot dengan garis tren.
- **Regresi** — uji VIF, model OLS (penduduk usia sekolah & luas wilayah → jumlah sekolah), interpretasi otomatis, plot residual, uji Breusch-Pagan, dan kalkulator prediksi interaktif.
- **Peta Sebaran** — peta interaktif lokasi sekolah (dengan sampling agar tetap ringan) dan peta bubble jumlah sekolah per provinsi.
- **Data** — tabel data terfilter dan data agregat provinsi, keduanya bisa diunduh sebagai CSV.

## Catatan
- Filter di sidebar (provinsi, jenjang, status) memengaruhi tab Overview, Peta, dan Data — tapi tab Regresi & EDA selalu memakai agregat 34 provinsi penuh, karena begitulah unit analisis modelnya dirancang.
- Kalau ingin pakai file data lain, tinggal upload lewat sidebar "Upload complete_data.csv (opsional)".
