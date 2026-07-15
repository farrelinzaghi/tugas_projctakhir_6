"""
Dashboard Analisis Persebaran Sekolah di Indonesia
====================================================
Aplikasi Streamlit untuk eksplorasi data, korelasi, dan regresi linear
antara penduduk usia sekolah, luas wilayah, dan jumlah sekolah per provinsi.

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Analisis Sekolah Indonesia",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "complete_data.csv"  # letakkan file ini satu folder dengan app.py

# ==============================================================================
# LOAD & AGGREGATE DATA
# ==============================================================================
@st.cache_data(show_spinner="Memuat data...")
def load_data(file):
    df_raw = pd.read_csv(file)
    return df_raw


@st.cache_data(show_spinner="Mengagregasi data ke level provinsi...")
def aggregate_province(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_raw.groupby("province_name")
        .agg(
            jumlah_sekolah=("school_name", "count"),
            luas_wilayah=("province_area", "first"),
            total_penduduk=("total_population", "first"),
            penduduk_usia_sekolah=("total_education_age_population", "first"),
        )
        .reset_index()
        .rename(columns={"province_name": "provinsi"})
    )
    kolom_numerik = ["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah", "jumlah_sekolah"]
    for col in kolom_numerik:
        df = df[df[col] >= 0]
    return df


@st.cache_data(show_spinner=False)
def fit_ols(df: pd.DataFrame):
    X = sm.add_constant(df[["penduduk_usia_sekolah", "luas_wilayah"]])
    y = df["jumlah_sekolah"]
    model = sm.OLS(y, X).fit()
    return model, X, y


@st.cache_data(show_spinner=False)
def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    X_check = df[["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah"]]
    X_check_const = sm.add_constant(X_check)
    vif_table = pd.DataFrame()
    vif_table["Variabel"] = X_check_const.columns
    vif_table["VIF"] = [
        variance_inflation_factor(X_check_const.values, i)
        for i in range(X_check_const.shape[1])
    ]
    return vif_table


# ==============================================================================
# SIDEBAR - SUMBER DATA & FILTER
# ==============================================================================
st.sidebar.title("⚙️ Pengaturan")

uploaded = st.sidebar.file_uploader("Upload complete_data.csv (opsional)", type=["csv"])

data_source = uploaded if uploaded is not None else CSV_PATH

try:
    df_raw = load_data(data_source)
except FileNotFoundError:
    st.error(
        f"File `{CSV_PATH}` tidak ditemukan di folder aplikasi. "
        "Silakan upload file CSV lewat sidebar di sebelah kiri."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data Sekolah (level sekolah)")

all_provinces = sorted(df_raw["province_name"].unique())
all_stages = sorted(df_raw["stage"].unique())
all_status = sorted(df_raw["status"].unique())

sel_provinces = st.sidebar.multiselect("Provinsi", all_provinces, default=[])
sel_stages = st.sidebar.multiselect("Jenjang (stage)", all_stages, default=[])
sel_status = st.sidebar.multiselect("Status (N=Negeri, S=Swasta)", all_status, default=[])

df_filtered = df_raw.copy()
if sel_provinces:
    df_filtered = df_filtered[df_filtered["province_name"].isin(sel_provinces)]
if sel_stages:
    df_filtered = df_filtered[df_filtered["stage"].isin(sel_stages)]
if sel_status:
    df_filtered = df_filtered[df_filtered["status"].isin(sel_status)]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Menampilkan **{len(df_filtered):,}** dari **{len(df_raw):,}** baris data sekolah."
)

# Data agregat provinsi selalu dihitung dari SELURUH data (bukan hasil filter sekolah),
# karena model regresi dirancang pada unit analisis provinsi.
df_prov = aggregate_province(df_raw)
kolom_numerik = ["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah", "jumlah_sekolah"]

# ==============================================================================
# HEADER
# ==============================================================================
st.title("🏫 Dashboard Analisis Persebaran Sekolah di Indonesia")
st.markdown(
    "Eksplorasi hubungan antara **penduduk usia sekolah**, **luas wilayah**, "
    "dan **jumlah sekolah** di 34 provinsi Indonesia."
)

tab_overview, tab_eda, tab_regresi, tab_peta, tab_data = st.tabs(
    ["📌 Overview", "📊 EDA & Korelasi", "📈 Regresi", "🗺️ Peta Sebaran", "🧾 Data"]
)

# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Ringkasan Nasional")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sekolah (raw)", f"{len(df_raw):,}")
    col2.metric("Jumlah Provinsi", f"{df_raw['province_name'].nunique()}")
    col3.metric("Total Penduduk Usia Sekolah", f"{df_prov['penduduk_usia_sekolah'].sum():,.0f}")
    col4.metric("Total Luas Wilayah (km²)", f"{df_prov['luas_wilayah'].sum():,.0f}")

    st.markdown("### Jumlah Sekolah per Provinsi")
    fig_bar = px.bar(
        df_prov.sort_values("jumlah_sekolah", ascending=False),
        x="provinsi",
        y="jumlah_sekolah",
        color="jumlah_sekolah",
        color_continuous_scale="Blues",
        labels={"jumlah_sekolah": "Jumlah Sekolah", "provinsi": "Provinsi"},
    )
    fig_bar.update_layout(xaxis_tickangle=-60, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Komposisi Jenjang Sekolah")
        stage_counts = df_filtered["stage"].value_counts().reset_index()
        stage_counts.columns = ["stage", "jumlah"]
        fig_pie = px.pie(stage_counts, names="stage", values="jumlah", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.markdown("### Komposisi Status Sekolah")
        status_counts = df_filtered["status"].value_counts().reset_index()
        status_counts.columns = ["status", "jumlah"]
        status_counts["status"] = status_counts["status"].map({"N": "Negeri", "S": "Swasta"})
        fig_pie2 = px.pie(status_counts, names="status", values="jumlah", hole=0.4)
        st.plotly_chart(fig_pie2, use_container_width=True)

# ==============================================================================
# TAB 2: EDA & KORELASI
# ==============================================================================
with tab_eda:
    st.subheader("Statistik Deskriptif (Level Provinsi, n=34)")
    deskriptif = df_prov[kolom_numerik].describe().T[["mean", "std", "min", "50%", "max"]]
    deskriptif.columns = ["Rata-rata", "Std Deviasi", "Minimum", "Median", "Maksimum"]
    st.dataframe(deskriptif.style.format("{:,.2f}"), use_container_width=True)

    st.subheader("Matriks Korelasi Pearson")
    corr_matrix = df_prov[kolom_numerik].corr(method="pearson")
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    c1 = corr_matrix.loc["total_penduduk", "penduduk_usia_sekolah"]
    c2 = corr_matrix.loc["penduduk_usia_sekolah", "jumlah_sekolah"]
    c3 = corr_matrix.loc["luas_wilayah", "jumlah_sekolah"]
    st.info(
        f"""
**Kesimpulan Korelasi**
- `total_penduduk` vs `penduduk_usia_sekolah`: **{c1:.2f}** (berpotensi multikolinieritas).
- `penduduk_usia_sekolah` vs `jumlah_sekolah`: **{c2:.2f}**
- `luas_wilayah` vs `jumlah_sekolah`: **{c3:.2f}**
"""
    )

    st.subheader("Scatter: Penduduk Usia Sekolah vs Jumlah Sekolah")
    fig_scatter = px.scatter(
        df_prov,
        x="penduduk_usia_sekolah",
        y="jumlah_sekolah",
        text="provinsi",
        trendline="ols",
        labels={"penduduk_usia_sekolah": "Penduduk Usia Sekolah", "jumlah_sekolah": "Jumlah Sekolah"},
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 3: REGRESI
# ==============================================================================
with tab_regresi:
    st.subheader("Uji Multikolinieritas (VIF)")
    vif_table = compute_vif(df_prov)
    st.dataframe(vif_table.style.format({"VIF": "{:.2f}"}), use_container_width=True)
    st.caption(
        "Variabel `total_penduduk` di-drop dari model karena redundan dengan "
        "`penduduk_usia_sekolah` (VIF tinggi / gejala multikolinieritas)."
    )

    st.subheader("Model Regresi Linear Berganda (OLS)")
    model, X_final, y = fit_ols(df_prov)

    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared", f"{model.rsquared:.4f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
    col3.metric("F-test p-value", f"{model.f_pvalue:.4g}")

    with st.expander("📄 Lihat Ringkasan Model Lengkap (statsmodels summary)"):
        st.text(model.summary().as_text())

    st.subheader("Interpretasi Otomatis")
    p_x1 = model.pvalues["penduduk_usia_sekolah"]
    p_x2 = model.pvalues["luas_wilayah"]
    f_p = model.f_pvalue
    r2 = model.rsquared

    kesimpulan_f = (
        "Model signifikan secara simultan, H0 ditolak."
        if f_p < 0.05
        else "Model TIDAK signifikan secara simultan pada alpha 5%, H0 gagal ditolak."
    )
    sig_x1 = "signifikan" if p_x1 < 0.05 else "tidak signifikan"
    sig_x2 = "signifikan" if p_x2 < 0.05 else "tidak signifikan"

    st.success(
        f"""
1. **Uji F simultan** (p-value = {f_p:.4g}): {kesimpulan_f}
2. **penduduk_usia_sekolah**: p-value = {p_x1:.4g} → **{sig_x1}** pada alpha 5%
3. **luas_wilayah**: p-value = {p_x2:.4g} → **{sig_x2}** pada alpha 5%
4. **R-squared** = {r2:.4f} → model menjelaskan **{r2*100:.1f}%** variasi jumlah sekolah antar provinsi.
"""
    )

    st.subheader("Plot Residual vs Fitted Values (Uji Homoskedastisitas)")
    df_prov_pred = df_prov.copy()
    df_prov_pred["prediksi"] = model.predict(X_final)
    df_prov_pred["residual"] = y - df_prov_pred["prediksi"]

    fig_resid = px.scatter(
        df_prov_pred,
        x="prediksi",
        y="residual",
        text="provinsi",
        labels={"prediksi": "Nilai Prediksi Jumlah Sekolah", "residual": "Residual (Error)"},
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_resid, use_container_width=True)

    bp_test = het_breuschpagan(model.resid, X_final)
    labels = ["LM Statistic", "LM p-value", "F-Statistic", "F p-value"]
    bp_df = pd.DataFrame({"Uji": labels, "Nilai": [f"{v:.4f}" for v in bp_test]})
    st.markdown("**Uji Breusch-Pagan (Homoskedastisitas)**")
    st.dataframe(bp_df, use_container_width=True, hide_index=True)

    st.subheader("Prediksi Interaktif")
    st.caption("Coba masukkan nilai untuk memprediksi jumlah sekolah berdasarkan model.")
    pc1, pc2 = st.columns(2)
    input_penduduk = pc1.number_input(
        "Penduduk usia sekolah", min_value=0,
        value=int(df_prov["penduduk_usia_sekolah"].median()), step=1000,
    )
    input_luas = pc2.number_input(
        "Luas wilayah (km²)", min_value=0,
        value=int(df_prov["luas_wilayah"].median()), step=100,
    )
    pred_input = pd.DataFrame({"const": [1], "penduduk_usia_sekolah": [input_penduduk], "luas_wilayah": [input_luas]})
    pred_value = model.predict(pred_input)[0]
    st.metric("Estimasi Jumlah Sekolah", f"{pred_value:,.0f}")

# ==============================================================================
# TAB 4: PETA SEBARAN
# ==============================================================================
with tab_peta:
    st.subheader("Peta Sebaran Lokasi Sekolah")

    df_map = df_filtered.dropna(subset=["lat", "long"])
    max_points = st.slider(
        "Jumlah titik ditampilkan (sampling agar peta tetap ringan)",
        min_value=1000, max_value=min(50000, max(len(df_map), 1000)),
        value=min(10000, len(df_map)) if len(df_map) > 0 else 1000, step=1000,
    )
    if len(df_map) > max_points:
        df_map_sample = df_map.sample(max_points, random_state=42)
    else:
        df_map_sample = df_map

    st.caption(f"Menampilkan {len(df_map_sample):,} dari {len(df_map):,} sekolah (dengan koordinat valid).")

    if len(df_map_sample) > 0:
        fig_map = px.scatter_mapbox(
            df_map_sample,
            lat="lat",
            lon="long",
            color="stage",
            hover_name="school_name",
            hover_data=["province_name", "city_name", "status"],
            zoom=3.3,
            height=650,
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Tidak ada data dengan koordinat valid untuk ditampilkan sesuai filter saat ini.")

    st.subheader("Jumlah Sekolah per Provinsi (Choropleth sederhana via bubble)")
    prov_coords = (
        df_raw.dropna(subset=["lat", "long"])
        .groupby("province_name")[["lat", "long"]]
        .mean()
        .reset_index()
        .merge(df_prov, left_on="province_name", right_on="provinsi")
    )
    fig_bubble = px.scatter_mapbox(
        prov_coords,
        lat="lat",
        lon="long",
        size="jumlah_sekolah",
        color="jumlah_sekolah",
        hover_name="provinsi",
        color_continuous_scale="Reds",
        zoom=3,
        height=600,
    )
    fig_bubble.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_bubble, use_container_width=True)

# ==============================================================================
# TAB 5: DATA MENTAH
# ==============================================================================
with tab_data:
    st.subheader("Data Sekolah (sesuai filter)")
    st.dataframe(df_filtered, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Unduh Data Terfilter (CSV)",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="data_sekolah_terfilter.csv",
        mime="text/csv",
    )

    st.subheader("Data Agregat per Provinsi")
    st.dataframe(df_prov, use_container_width=True, height=400)
    st.download_button(
        "⬇️ Unduh Data Agregat Provinsi (CSV)",
        df_prov.to_csv(index=False).encode("utf-8"),
        file_name="data_agregat_provinsi.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Dibuat dengan Streamlit • Data sekolah se-Indonesia diagregasi ke level provinsi.")
