# ---
# Pertemuan 4: Data Cleaning dan Quality di Skala Besar
# Kuliah Analisis Big Data - Semester Ganjil 2026
# ---
#
# Notebook ini adalah panduan praktik untuk Pertemuan 4.
# Jalankan cell-cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 4: Data Cleaning dan Quality di Skala Besar
#
# ## Tujuan Pembelajaran
# 1. Memahami 6 Dimensi Kualitas Data dan cara mengukurnya
# 2. Melakukan profiling data cepat dengan DuckDB SUMMARIZE
# 3. Menangani missing values, outliers, dan duplikat dengan Polars
# 4. Menulis validation rules dasar dengan Great Expectations
# 5. Membangun pipeline data cleaning yang reproducible dan terdokumentasi

# %% [markdown]
# ---
# ## Bagian 1: Setup dan Pembuatan Data "Kotor" (Dirty Data)
#
# Kita akan mensimulasikan dataset e-commerce Indonesia yang memiliki berbagai
# masalah kualitas data yang umum ditemui di industri.

# %%
from pathlib import Path

import polars as pl
import duckdb
import numpy as np
import os

repo_root = Path(__file__).resolve().parents[2]
temp_dir = repo_root / ".tmp_data"
temp_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
N = 500_000
print(f"Membuat dataset dengan masalah kualitas data ({N:,} baris)...")

# Data dasar yang bersih
kota_list = ["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang"]
kategori_list = ["Elektronik", "Fashion", "Makanan", "Kesehatan", "Olahraga"]

# Membuat data kotor dengan berbagai masalah sengaja
id_arr = list(range(N))
# Sisipkan 5% duplikat
duplikat_idx = np.random.choice(N, size=int(N * 0.05), replace=False)

tanggal_base = pl.datetime_range(
    pl.datetime(2024, 1, 1),
    pl.datetime(2024, 12, 31),
    interval="1h",
    eager=True,
)
tanggal_arr = tanggal_base.sample(N, with_replacement=True, seed=42)

harga_arr = np.random.randint(10_000, 5_000_000, N).astype(float)
# Sisipkan outlier ekstrem (1%)
outlier_idx = np.random.choice(N, size=int(N * 0.01), replace=False)
harga_arr[outlier_idx] = np.random.choice([0, -100, 999_999_999, -1], len(outlier_idx))

kuantitas_arr = np.random.randint(1, 50, N).astype(float)
kuantitas_arr[np.random.choice(N, size=int(N * 0.03), replace=False)] = None  # 3% null

rating_arr = np.random.uniform(1.0, 5.0, N)
rating_arr[np.random.choice(N, size=int(N * 0.02), replace=False)] = np.nan  # 2% null

nama_kota = np.random.choice(kota_list, N)
# Variasi penulisan kota (inkonsistensi)
inkonsisten = np.random.choice(N, size=int(N * 0.04), replace=False)
for idx in inkonsisten[:len(inkonsisten)//3]:
    nama_kota[idx] = nama_kota[idx].upper()
for idx in inkonsisten[len(inkonsisten)//3:2*len(inkonsisten)//3]:
    nama_kota[idx] = nama_kota[idx].lower()
for idx in inkonsisten[2*len(inkonsisten)//3:]:
    nama_kota[idx] = " " + nama_kota[idx] + " "  # spasi ekstra

email_arr = [f"user{i}@{'gmail' if i%3==0 else 'yahoo' if i%3==1 else 'outlook'}.com" for i in range(N)]
# Email tidak valid untuk sebagian
email_invalid_idx = np.random.choice(N, size=int(N * 0.03), replace=False)
for idx in email_invalid_idx:
    email_arr[idx] = np.random.choice(["bukan-email", "abc@", "@domain.com", ""])

df_kotor = pl.DataFrame({
    "id_transaksi": id_arr,
    "tanggal": tanggal_arr,
    "kota": nama_kota.tolist(),
    "kategori": np.random.choice(kategori_list, N).tolist(),
    "harga": harga_arr.tolist(),
    "kuantitas": kuantitas_arr.tolist(),
    "rating": rating_arr.tolist(),
    "email_pembeli": email_arr,
    "status": np.random.choice(["selesai", "proses", "batal", None], N,
                                p=[0.6, 0.15, 0.2, 0.05]).tolist(),
})

# Tambahkan duplikat
baris_duplikat = df_kotor[duplikat_idx]
df_kotor = pl.concat([df_kotor, baris_duplikat])

print(f"Dataset dibuat: {df_kotor.shape[0]:,} baris x {df_kotor.shape[1]} kolom")
print(f"(termasuk {len(duplikat_idx):,} baris duplikat yang disengaja)")

# %% [markdown]
# ---
# ## Bagian 2: Profiling Data dengan DuckDB SUMMARIZE
#
# Langkah pertama dalam data cleaning adalah **memahami data** melalui profiling.
# DuckDB SUMMARIZE memberikan statistik lengkap untuk setiap kolom dalam satu perintah.

# %%
con = duckdb.connect()
con.register("transaksi", df_kotor)

print("=== DuckDB SUMMARIZE: Profiling Cepat ===")
try:
    profil = con.execute("SUMMARIZE SELECT * FROM transaksi").df()
except Exception:
    profil = con.execute("""
        SELECT
            column_name AS column_name,
            data_type AS column_type,
            NULL AS min,
            NULL AS max,
            NULL AS avg,
            NULL AS std
        FROM information_schema.columns
        WHERE table_name = 'transaksi'
        ORDER BY column_name
    """).df()
print(profil.to_string())

# %%
# Hitung masalah kualitas data secara menyeluruh
print("\n=== Laporan Kualitas Data ===")
laporan = con.execute("""
    SELECT
        COUNT(*) AS total_baris,
        COUNT(DISTINCT id_transaksi) AS id_unik,
        COUNT(*) - COUNT(DISTINCT id_transaksi) AS duplikat,

        -- Missing values per kolom
        SUM(CASE WHEN harga IS NULL THEN 1 ELSE 0 END) AS null_harga,
        SUM(CASE WHEN kuantitas IS NULL THEN 1 ELSE 0 END) AS null_kuantitas,
        SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS null_rating,
        SUM(CASE WHEN status IS NULL THEN 1 ELSE 0 END) AS null_status,
        SUM(CASE WHEN email_pembeli IS NULL OR email_pembeli = '' THEN 1 ELSE 0 END) AS null_email,

        -- Nilai tidak valid
        SUM(CASE WHEN harga <= 0 THEN 1 ELSE 0 END) AS harga_invalid,
        SUM(CASE WHEN rating < 1 OR rating > 5 THEN 1 ELSE 0 END) AS rating_invalid,
        SUM(CASE WHEN harga > 100_000_000 THEN 1 ELSE 0 END) AS harga_outlier_ekstrem
    FROM transaksi
""").df()

for kolom in laporan.columns:
    nilai = laporan[kolom].iloc[0]
    persen = nilai / laporan["total_baris"].iloc[0] * 100 if kolom != "total_baris" else 100
    if nilai > 0:
        print(f"  {kolom:<30}: {nilai:>8,}  ({persen:5.1f}%)")

# %% [markdown]
# ---
# ## Bagian 3: Menangani Missing Values
#
# Strategi penanganan missing values bergantung pada:
# - Persentase missing (sedikit vs banyak)
# - Jenis data (numerik, kategorik, tanggal)
# - Dampak pada analisis

# %%
# Lihat distribusi missing values
print("=== Distribusi Missing Values ===")
missing_summary = {
    kolom: {
        "jumlah_null": df_kotor[kolom].null_count(),
        "persen_null": round(df_kotor[kolom].null_count() / len(df_kotor) * 100, 2)
    }
    for kolom in df_kotor.columns
}
for kolom, info in missing_summary.items():
    if info["jumlah_null"] > 0:
        print(f"  {kolom:<20}: {info['jumlah_null']:>6,} null  ({info['persen_null']:.1f}%)")

# %%
# Strategi penanganan missing values dengan Polars
print("\n=== Strategi Penanganan Missing Values ===")

df_cleaned_mv = df_kotor.with_columns([
    # 1. Kuantitas (numerik, 3% null): isi dengan median
    pl.col("kuantitas").fill_null(
        pl.col("kuantitas").median()
    ).alias("kuantitas"),

    # 2. Rating (numerik, 2% null): isi dengan median per kategori
    pl.col("rating").fill_null(
        pl.col("rating").mean().over("kategori")
    ).round(1).alias("rating"),

    # 3. Status (kategorik, 5% null): isi dengan modus ("proses" adalah default)
    pl.col("status").fill_null("proses").alias("status"),
])

print("Sebelum penanganan:")
for kolom in ["kuantitas", "rating", "status"]:
    print(f"  {kolom}: {df_kotor[kolom].null_count():,} null")

print("\nSetelah penanganan:")
for kolom in ["kuantitas", "rating", "status"]:
    print(f"  {kolom}: {df_cleaned_mv[kolom].null_count():,} null")

# %% [markdown]
# ---
# ## Bagian 4: Menangani Outliers
#
# Outlier bisa menjadi:
# - **Kesalahan data**: nilai tidak mungkin secara bisnis (harga negatif)
# - **Nilai ekstrem nyata**: transaksi sangat besar yang valid
# - **Data anomali**: perlu investigasi lebih lanjut

# %%
# Deteksi outlier dengan metode IQR (Interquartile Range)
print("=== Deteksi Outlier dengan Metode IQR ===")

# Hitung statistik distribusi harga
statistik_harga = df_cleaned_mv.select([
    pl.col("harga").min().alias("min"),
    pl.col("harga").quantile(0.01).alias("p1"),
    pl.col("harga").quantile(0.25).alias("q1"),
    pl.col("harga").quantile(0.50).alias("median"),
    pl.col("harga").quantile(0.75).alias("q3"),
    pl.col("harga").quantile(0.99).alias("p99"),
    pl.col("harga").max().alias("max"),
    pl.col("harga").mean().alias("mean"),
])
print("Distribusi Harga:")
print(statistik_harga)

# %%
# Metode IQR untuk deteksi outlier
q1 = df_cleaned_mv["harga"].quantile(0.25)
q3 = df_cleaned_mv["harga"].quantile(0.75)
iqr = q3 - q1
batas_bawah_iqr = q1 - 1.5 * iqr
batas_atas_iqr = q3 + 1.5 * iqr

# Batas bisnis (domain knowledge): harga valid antara 1.000 dan 50.000.000
batas_bawah_bisnis = 1_000
batas_atas_bisnis = 50_000_000

print(f"\nMetode IQR:")
print(f"  Q1 = {q1:,.0f}, Q3 = {q3:,.0f}, IQR = {iqr:,.0f}")
print(f"  Batas bawah IQR: {batas_bawah_iqr:,.0f}")
print(f"  Batas atas IQR:  {batas_atas_iqr:,.0f}")
print(f"\nBatas Bisnis (domain knowledge):")
print(f"  Min valid: Rp {batas_bawah_bisnis:,}")
print(f"  Max valid: Rp {batas_atas_bisnis:,}")

# %%
# Tangani outlier: gabungkan IQR dengan domain knowledge
df_cleaned_outlier = df_cleaned_mv.with_columns([
    # Tandai outlier
    pl.when(
        (pl.col("harga") < batas_bawah_bisnis) |
        (pl.col("harga") > batas_atas_bisnis) |
        pl.col("harga").is_null()
    )
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("is_outlier_harga"),

    # Clamp harga ke batas yang valid (windsorization)
    pl.col("harga")
    .clip(lower_bound=batas_bawah_bisnis, upper_bound=batas_atas_bisnis)
    .alias("harga_clean"),
])

n_outlier = df_cleaned_outlier["is_outlier_harga"].sum()
print(f"\nJumlah outlier harga terdeteksi: {n_outlier:,} ({n_outlier/len(df_cleaned_outlier)*100:.1f}%)")

# Hapus baris dengan harga tidak valid (harga negatif atau nol)
df_cleaned_outlier = df_cleaned_outlier.filter(pl.col("harga") > 0)
print(f"Baris setelah hapus harga invalid: {len(df_cleaned_outlier):,}")

# %% [markdown]
# ---
# ## Bagian 5: Standardisasi dan Normalisasi Data
#
# Masalah inkonsistensi sangat umum pada data teks: huruf besar/kecil, spasi ekstra,
# typo, dan penulisan yang tidak seragam.

# %%
# Melihat inkonsistensi kota
print("=== Inkonsistensi Penulisan Kota ===")
variasi_kota = (
    df_cleaned_outlier
    .group_by("kota")
    .agg(pl.col("id_transaksi").count().alias("jumlah"))
    .sort("jumlah", descending=True)
)
print(f"Jumlah variasi penulisan unik: {len(variasi_kota)}")
print(variasi_kota.head(20))

# %%
# Standardisasi: bersihkan dan normalisasi kolom teks
print("\n=== Standardisasi Kolom Teks ===")
df_cleaned_teks = df_cleaned_outlier.with_columns([
    # Normalisasi kota: hapus spasi, Title Case
    pl.col("kota")
    .str.strip_chars()          # Hapus spasi di awal dan akhir
    .str.to_titlecase()         # Title Case (Huruf Pertama Kapital)
    .alias("kota"),

    # Normalisasi kategori
    pl.col("kategori")
    .str.strip_chars()
    .str.to_titlecase()
    .alias("kategori"),

    # Normalisasi status: lowercase
    pl.col("status")
    .str.strip_chars()
    .str.to_lowercase()
    .alias("status"),
])

print("Variasi kota setelah standardisasi:")
print(
    df_cleaned_teks
    .group_by("kota")
    .agg(pl.col("id_transaksi").count().alias("jumlah"))
    .sort("jumlah", descending=True)
)

# %% [markdown]
# ---
# ## Bagian 6: Validasi Email dan Format Data

# %%
# Validasi format email dengan regex
print("=== Validasi Format Email ===")

df_cleaned_email = df_cleaned_teks.with_columns([
    # Cek apakah email mengandung @ dan domain
    pl.col("email_pembeli")
    .str.contains(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
    .alias("email_valid"),
])

n_email_valid = df_cleaned_email["email_valid"].sum()
n_email_invalid = len(df_cleaned_email) - n_email_valid
print(f"Email valid:   {n_email_valid:,} ({n_email_valid/len(df_cleaned_email)*100:.1f}%)")
print(f"Email invalid: {n_email_invalid:,} ({n_email_invalid/len(df_cleaned_email)*100:.1f}%)")

# Tampilkan contoh email yang tidak valid
print("\nContoh email tidak valid:")
print(
    df_cleaned_email
    .filter(~pl.col("email_valid"))
    .select("email_pembeli")
    .unique()
    .head(10)
)

# Opsi: ganti email invalid dengan null (anonim)
df_cleaned_email = df_cleaned_email.with_columns(
    pl.when(pl.col("email_valid"))
    .then(pl.col("email_pembeli"))
    .otherwise(None)
    .alias("email_pembeli")
)

# %% [markdown]
# ---
# ## Bagian 7: Menangani Duplikat

# %%
# Identifikasi duplikat
print("=== Identifikasi Baris Duplikat ===")
total_baris = len(df_cleaned_email)
baris_unik = df_cleaned_email.n_unique()
baris_duplikat = total_baris - baris_unik

print(f"Total baris:      {total_baris:,}")
print(f"Baris unik:       {baris_unik:,}")
print(f"Baris duplikat:   {baris_duplikat:,} ({baris_duplikat/total_baris*100:.1f}%)")

# %%
# Hapus duplikat berdasarkan id_transaksi (pertahankan yang pertama)
print("\n=== Penghapusan Duplikat ===")
df_dedup = df_cleaned_email.unique(
    subset=["id_transaksi"],  # Unik berdasarkan kolom kunci
    keep="first",             # Pertahankan kemunculan pertama
    maintain_order=True
)
print(f"Baris setelah deduplikasi: {len(df_dedup):,}")
print(f"Baris terhapus: {total_baris - len(df_dedup):,}")

# %% [markdown]
# ---
# ## Bagian 8: Great Expectations — Validasi Data Otomatis
#
# Great Expectations (GX) memungkinkan kita mendefinisikan aturan validasi data
# secara deklaratif, sehingga kualitas data dapat dicek otomatis di pipeline.

# %%
try:
    import great_expectations as gx

    print(f"Great Expectations version: {gx.__version__}")

    # Buat context GX in-memory
    ctx = gx.get_context(mode="ephemeral")

    # Tambahkan datasource dari DataFrame Polars (via pandas bridge)
    ds = ctx.data_sources.add_pandas(name="transaksi_bersih")
    da = ds.add_dataframe_asset(name="transaksi")

    batch_def = da.add_batch_definition_whole_dataframe("seluruh_data")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df_dedup.to_pandas()})

    # Definisi expectations (aturan validasi)
    suite = ctx.suites.add(gx.ExpectationSuite(name="aturan_transaksi"))

    # Ekspektasi 1: harga tidak boleh null dan harus positif
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="harga_clean",
            min_value=1_000,
            max_value=50_000_000,
        )
    )
    # Ekspektasi 2: rating harus antara 1 dan 5
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="rating",
            min_value=1.0,
            max_value=5.0,
        )
    )
    # Ekspektasi 3: status harus salah satu dari nilai yang valid
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status",
            value_set=["selesai", "proses", "batal", "dikirim", "dikembalikan"],
        )
    )
    # Ekspektasi 4: id_transaksi harus unik
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="id_transaksi")
    )
    # Ekspektasi 5: tidak boleh ada null di kolom kritis
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="id_transaksi")
    )

    # Jalankan validasi
    vd = ctx.validation_definitions.add(
        gx.ValidationDefinition(name="validasi_utama", data=batch_def, suite=suite)
    )
    hasil_validasi = vd.run(batch_parameters={"dataframe": df_dedup.to_pandas()})

    # Tampilkan ringkasan hasil
    print("\n=== Hasil Validasi Great Expectations ===")
    print(f"Status: {'LULUS' if hasil_validasi.success else 'GAGAL'}")
    for ev in hasil_validasi.results:
        status = "OK" if ev.success else "GAGAL"
        expectation_type = ev.expectation_config.type
        print(f"  [{status}] {expectation_type}")

except ImportError:
    print("Great Expectations tidak terinstal. Install dengan: pip install great_expectations")
    print("Menampilkan validasi manual sebagai alternatif...")

    # Validasi manual menggunakan DuckDB (alternatif tanpa Great Expectations)
    con.register("transaksi_bersih", df_dedup)
    hasil_validasi_manual = con.execute("""
        SELECT
            'harga_valid' AS aturan,
            SUM(CASE WHEN harga_clean BETWEEN 1000 AND 50000000 THEN 0 ELSE 1 END) AS pelanggaran,
            COUNT(*) AS total
        FROM transaksi_bersih
        UNION ALL
        SELECT
            'rating_valid',
            SUM(CASE WHEN rating BETWEEN 1.0 AND 5.0 THEN 0 ELSE 1 END),
            COUNT(*)
        FROM transaksi_bersih
        UNION ALL
        SELECT
            'status_valid',
            SUM(CASE WHEN status NOT IN ('selesai','proses','batal','dikirim','dikembalikan')
                THEN 1 ELSE 0 END),
            COUNT(*)
        FROM transaksi_bersih
    """).df()
    print(hasil_validasi_manual)

# %% [markdown]
# ---
# ## Bagian 9: Pipeline Cleaning Lengkap (Chained Operations)
#
# Di industri, langkah cleaning digabung menjadi satu pipeline yang bersih,
# terdokumentasi, dan dapat direproduksi.

# %%
def pipeline_data_cleaning(df_input: pl.DataFrame) -> pl.DataFrame:
    """
    Pipeline data cleaning lengkap untuk dataset transaksi e-commerce.

    Langkah:
    1. Hapus duplikat berdasarkan id_transaksi
    2. Hapus harga yang tidak valid (<= 0)
    3. Isi missing values dengan strategi yang sesuai
    4. Standardisasi teks (kota, kategori, status)
    5. Validasi dan tandai email tidak valid
    6. Tambahkan kolom derivasi (total_bayar, dll.)
    """
    return (
        df_input
        # Langkah 1: Deduplikasi
        .unique(subset=["id_transaksi"], keep="first", maintain_order=True)
        # Langkah 2: Hapus harga tidak valid
        .filter(pl.col("harga").is_not_null() & (pl.col("harga") > 0))
        # Langkah 3: Isi missing values
        .with_columns([
            pl.col("kuantitas").fill_null(pl.col("kuantitas").median()),
            pl.col("rating").fill_null(pl.col("rating").mean().over("kategori")).round(1),
            pl.col("status").fill_null("proses"),
        ])
        # Langkah 4: Standardisasi teks
        .with_columns([
            pl.col("kota").str.strip_chars().str.to_titlecase(),
            pl.col("kategori").str.strip_chars().str.to_titlecase(),
            pl.col("status").str.strip_chars().str.to_lowercase(),
        ])
        # Langkah 5: Clamp harga ke batas bisnis
        .with_columns([
            pl.col("harga").clip(lower_bound=1_000, upper_bound=50_000_000).alias("harga"),
        ])
        # Langkah 6: Kolom derivasi
        .with_columns([
            (pl.col("harga") * pl.col("kuantitas")).alias("total_bayar"),
            pl.when(pl.col("status") == "selesai")
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("transaksi_selesai"),
        ])
    )


# Jalankan pipeline
print("=== Menjalankan Pipeline Cleaning ===")
df_bersih = pipeline_data_cleaning(df_kotor)

print(f"Sebelum cleaning: {len(df_kotor):,} baris")
print(f"Setelah cleaning:  {len(df_bersih):,} baris")
print(f"Baris dihapus:     {len(df_kotor) - len(df_bersih):,}")

# Simpan hasil bersih
output_dir = temp_dir / "transaksi_bersih"
output_dir.mkdir(parents=True, exist_ok=True)
parquet_output = output_dir / "transaksi_bersih.parquet"
df_bersih.write_parquet(parquet_output)
print(f"\nData bersih tersimpan ke: {parquet_output}")
print(df_bersih.head(5))

# %%
# Laporan akhir kualitas data
print("\n=== Laporan Akhir Kualitas Data ===")
print(f"{'Metrik':<35} {'Sebelum':>12} {'Setelah':>12} {'Perbaikan':>12}")
print("-" * 74)
metrik = [
    ("Total Baris", len(df_kotor), len(df_bersih)),
    ("Baris Duplikat", len(df_kotor) - df_kotor.n_unique(), len(df_bersih) - df_bersih.n_unique()),
    ("Null Kuantitas", df_kotor["kuantitas"].null_count(), df_bersih["kuantitas"].null_count()),
    ("Null Rating", df_kotor["rating"].null_count(), df_bersih["rating"].null_count()),
    ("Null Status", df_kotor["status"].null_count(), df_bersih["status"].null_count()),
    ("Harga <= 0", (df_kotor["harga"] <= 0).sum(), (df_bersih["harga"] <= 0).sum()),
]
for label, sebelum, sesudah in metrik:
    perbaikan = sebelum - sesudah
    print(f"  {label:<33} {sebelum:>12,} {sesudah:>12,} {perbaikan:>+12,}")

# %% [markdown]
# ---
# ## Ringkasan Pertemuan 4
#
# Hari ini kita telah mempelajari:
# 1. 6 Dimensi Kualitas Data: Completeness, Accuracy, Consistency, Timeliness, Uniqueness, Validity
# 2. Profiling cepat dengan DuckDB SUMMARIZE: mendapat gambaran menyeluruh dalam satu perintah
# 3. Strategi missing values: hapus, isi median/mean/modus, atau isi berdasarkan konteks grup
# 4. Deteksi & penanganan outlier: metode IQR + domain knowledge + windsorization
# 5. Standardisasi teks: strip, title case, lowercase, regex validation
# 6. Deduplikasi: unique() dengan subset kunci dan strategi keep
# 7. Great Expectations: framework validasi data otomatis untuk pipeline produksi
# 8. Pipeline cleaning yang modular, terdokumentasi, dan dapat direproduksi
#
# ### Pertemuan Berikutnya
# Exploratory Data Analysis (EDA) & Visualisasi Interaktif — Analisis mendalam
# dengan pipeline DuckDB -> Polars -> Plotly/Altair dan visualisasi peta Indonesia
