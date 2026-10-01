# ---
# Pertemuan 3: SQL untuk Big Data — DuckDB dan Format Data
# Kuliah Analisis Big Data - Semester Ganjil 2026
# ---
#
# Notebook ini adalah panduan praktik untuk Pertemuan 3.
# Jalankan cell-cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 3: SQL untuk Big Data — DuckDB dan Format Data
#
# ## Tujuan Pembelajaran
# 1. Memahami DuckDB sebagai mesin SQL analitik in-process (OLAP)
# 2. Menggunakan DuckDB untuk query langsung ke file CSV, Parquet, dan JSON
# 3. Memahami teknik optimasi: Predicate Pushdown dan Column Pruning
# 4. Menguasai Window Functions SQL pada data besar
# 5. Memahami interoperabilitas DuckDB <-> Polars melalui Apache Arrow

# %% [markdown]
# ---
# ## Bagian 1: Persiapan — DuckDB dan Dataset

# %%
from pathlib import Path

import duckdb
import polars as pl
import numpy as np
import time
import os
import json

repo_root = Path(__file__).resolve().parents[2]
temp_dir = repo_root / ".tmp_data"
temp_dir.mkdir(parents=True, exist_ok=True)

# Buat koneksi DuckDB (in-memory database)
con = duckdb.connect()

print(f"DuckDB version: {duckdb.__version__}")
print(f"Polars version: {pl.__version__}")

# %% [markdown]
# ### 1a. Membuat Dataset Simulasi
#
# Kita akan membuat dataset transaksi e-government / layanan publik simulasi Indonesia.

# %%
# Generate dataset simulasi
np.random.seed(2026)
N = 1_500_000

layanan_list = [
    "KTP", "KK", "Akta Lahir", "SIM", "STNK", "Paspor",
    "NPWP", "BPJS", "Sertifikat Tanah", "IMB"
]
provinsi_list = [
    "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Banten",
    "Sumatera Utara", "Sumatera Selatan", "Kalimantan Timur", "Sulawesi Selatan", "Bali"
]
status_list = ["selesai", "proses", "antrian", "ditolak", "batal"]

data = {
    "id_layanan": range(N),
    "tanggal": pl.datetime_range(
        pl.datetime(2023, 1, 1), pl.datetime(2024, 12, 31), interval="1h",
        eager=True
    ).sample(N, with_replacement=True, seed=2026),
    "jenis_layanan": np.random.choice(layanan_list, N),
    "provinsi": np.random.choice(provinsi_list, N),
    "biaya": np.random.choice(
        [0, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 350_000], N
    ),
    "durasi_menit": np.abs(np.random.normal(45, 20, N)).astype(int) + 5,
    "status": np.random.choice(status_list, N, p=[0.6, 0.15, 0.1, 0.08, 0.07]),
    "kepuasan": np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.10, 0.20, 0.35, 0.30]),
    "online": np.random.choice([True, False], N, p=[0.45, 0.55]),
}

df = pl.DataFrame(data)

# Simpan ke berbagai format
layanan_dir = temp_dir / "layanan_publik"
layanan_dir.mkdir(parents=True, exist_ok=True)
parquet_path = layanan_dir / "layanan.parquet"
csv_path = layanan_dir / "layanan.csv"
json_path = layanan_dir / "layanan_sample.json"

df.write_parquet(parquet_path)
df.write_csv(csv_path)
# Simpan 1000 baris sebagai JSON (JSON cocok untuk sample kecil)
sample_json = (
    df.head(1_000)
    .with_columns(pl.col("tanggal").dt.to_string("%Y-%m-%d %H:%M:%S").alias("tanggal"))
    .to_dicts()
)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(sample_json, f)

print(f"Dataset dibuat: {N:,} baris")
print(f"  Parquet: {os.path.getsize(parquet_path)/1e6:.1f} MB")
print(f"  CSV:     {os.path.getsize(csv_path)/1e6:.1f} MB")
print(f"  JSON:    {os.path.getsize(json_path)/1e3:.1f} KB (sample 1000 baris)")

# %% [markdown]
# ---
# ## Bagian 2: DuckDB — Query Langsung ke File
#
# Kemampuan luar biasa DuckDB: query file Parquet, CSV, JSON **tanpa** import data
# ke database terlebih dahulu.

# %%
# Query langsung ke file Parquet
print("=== Query Langsung ke Parquet ===")
hasil = con.execute(f"""
    SELECT
        jenis_layanan,
        COUNT(*) AS jumlah,
        AVG(durasi_menit) AS rata_durasi,
        AVG(kepuasan) AS rata_kepuasan,
        SUM(biaya) AS total_pendapatan
    FROM '{parquet_path}'
    WHERE status = 'selesai'
    GROUP BY jenis_layanan
    ORDER BY jumlah DESC
""").df()
print(hasil)

# %%
# Query langsung ke CSV (tanpa membaca seluruh file ke memori)
print("\n=== Query Langsung ke CSV ===")
hasil_csv = con.execute(f"""
    SELECT
        provinsi,
        COUNT(*) AS total_permohonan,
        SUM(CASE WHEN status = 'selesai' THEN 1 ELSE 0 END) AS selesai,
        ROUND(SUM(CASE WHEN online THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS persen_online
    FROM '{csv_path}'
    GROUP BY provinsi
    ORDER BY total_permohonan DESC
""").df()
print(hasil_csv)

# %%
# Query langsung ke JSON
print("\n=== Query Langsung ke JSON ===")
hasil_json = con.execute(f"""
    SELECT
        jenis_layanan,
        COUNT(*) AS jumlah,
        AVG(durasi_menit) AS rata_durasi
    FROM '{json_path}'
    GROUP BY jenis_layanan
    ORDER BY jumlah DESC
""").df()
print(hasil_json)

# %% [markdown]
# ---
# ## Bagian 3: Predicate Pushdown dan Column Pruning
#
# DuckDB secara otomatis mengoptimalkan query dengan:
# - **Predicate Pushdown**: Filter `WHERE` diterapkan saat membaca file, bukan setelahnya
# - **Column Pruning (Projection Pushdown)**: Hanya kolom yang diperlukan yang dibaca

# %%
# Demonstrasi efek predicate pushdown
print("=== Benchmark Predicate Pushdown ===")

# Tanpa pushdown: baca semua dulu, filter kemudian (simulasi)
start = time.perf_counter()
df_all = pl.read_parquet(parquet_path)
df_filtered = df_all.filter(pl.col("provinsi") == "DKI Jakarta")
waktu_tanpa_pushdown = time.perf_counter() - start
print(f"Tanpa pushdown (read all + filter): {waktu_tanpa_pushdown:.4f} detik")
del df_all, df_filtered

# Dengan pushdown: DuckDB filter saat baca file
start = time.perf_counter()
hasil_pd = con.execute(f"""
    SELECT *
    FROM '{parquet_path}'
    WHERE provinsi = 'DKI Jakarta'
""").pl()
waktu_dengan_pushdown = time.perf_counter() - start
print(f"Dengan pushdown (DuckDB):           {waktu_dengan_pushdown:.4f} detik")
print(f"Speedup: {waktu_tanpa_pushdown/waktu_dengan_pushdown:.1f}x")
print(f"Baris hasil: {len(hasil_pd):,}")

# %%
# Column Pruning: hanya baca kolom yang dibutuhkan
print("\n=== Benchmark Column Pruning ===")

# Tanpa pruning: baca semua kolom
start = time.perf_counter()
_ = con.execute(f"SELECT * FROM '{parquet_path}'").pl()
waktu_semua_kolom = time.perf_counter() - start

# Dengan pruning: hanya 2 kolom
start = time.perf_counter()
_ = con.execute(f"SELECT jenis_layanan, biaya FROM '{parquet_path}'").pl()
waktu_dua_kolom = time.perf_counter() - start

print(f"Semua kolom ({df.width} kolom): {waktu_semua_kolom:.4f} detik")
print(f"Dua kolom saja:              {waktu_dua_kolom:.4f} detik")
print(f"Speedup: {waktu_semua_kolom/waktu_dua_kolom:.1f}x")

# %% [markdown]
# ---
# ## Bagian 4: Window Functions SQL
#
# Window Functions adalah salah satu fitur SQL yang paling powerful untuk analitik.
# Berbeda dengan GROUP BY, window functions tidak mereduksi jumlah baris.

# %%
# Contoh 1: RANK dan ROW_NUMBER
print("=== Window Function: RANK per Provinsi ===")
hasil_rank = con.execute(f"""
    WITH ringkasan AS (
        SELECT
            provinsi,
            jenis_layanan,
            COUNT(*) AS jumlah,
            AVG(kepuasan) AS rata_kepuasan
        FROM '{parquet_path}'
        WHERE status = 'selesai'
        GROUP BY provinsi, jenis_layanan
    )
    SELECT
        provinsi,
        jenis_layanan,
        jumlah,
        ROUND(rata_kepuasan, 2) AS rata_kepuasan,
        RANK() OVER (PARTITION BY provinsi ORDER BY jumlah DESC) AS rank_layanan,
        SUM(jumlah) OVER (PARTITION BY provinsi) AS total_provinsi,
        ROUND(jumlah * 100.0 / SUM(jumlah) OVER (PARTITION BY provinsi), 1) AS persen
    FROM ringkasan
    QUALIFY rank_layanan <= 3
    ORDER BY provinsi, rank_layanan
""").df()
print(hasil_rank.head(20))

# %%
# Contoh 2: Running total dan LAG/LEAD
print("\n=== Window Function: Running Total per Bulan ===")
hasil_running = con.execute(f"""
    WITH bulanan AS (
        SELECT
            EXTRACT(YEAR FROM tanggal) AS tahun,
            EXTRACT(MONTH FROM tanggal) AS bulan,
            COUNT(*) AS jumlah_permohonan,
            SUM(biaya) AS total_pendapatan
        FROM '{parquet_path}'
        WHERE status = 'selesai'
        GROUP BY 1, 2
    )
    SELECT
        tahun,
        bulan,
        jumlah_permohonan,
        total_pendapatan,
        SUM(jumlah_permohonan) OVER (PARTITION BY tahun ORDER BY bulan) AS kumulatif_permohonan,
        LAG(jumlah_permohonan) OVER (PARTITION BY tahun ORDER BY bulan) AS bulan_sebelumnya,
        ROUND(
            (jumlah_permohonan - LAG(jumlah_permohonan) OVER (PARTITION BY tahun ORDER BY bulan))
            * 100.0
            / NULLIF(LAG(jumlah_permohonan) OVER (PARTITION BY tahun ORDER BY bulan), 0),
        1) AS persen_pertumbuhan
    FROM bulanan
    ORDER BY tahun, bulan
""").df()
print(hasil_running)

# %% [markdown]
# ---
# ## Bagian 5: Partitioned Parquet
#
# Menyimpan data dalam partisi memungkinkan DuckDB untuk hanya membaca
# file yang relevan dengan query, bukan seluruh dataset.

# %%
# Simpan data dengan partisi per provinsi dan jenis layanan
print("=== Membuat Partitioned Parquet ===")
partisi_dir = temp_dir / "layanan_partisi"

(
    df.with_columns([
        pl.col("tanggal").dt.year().alias("tahun"),
    ])
    .write_parquet(
        partisi_dir,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": ["tahun", "provinsi"]},
    )
)
print(f"Data terpartisi tersimpan di: {partisi_dir}")

# Hitung jumlah file
jumlah_file = sum(1 for _, _, files in os.walk(partisi_dir) for f in files if f.endswith(".parquet"))
print(f"Jumlah file Parquet: {jumlah_file}")

# %%
# Query pada data terpartisi — hanya baca partisi yang relevan
print("\n=== Query pada Data Terpartisi (hive-style) ===")
start = time.perf_counter()
hasil_partisi = con.execute(f"""
    SELECT
        jenis_layanan,
        COUNT(*) AS jumlah,
        AVG(kepuasan) AS rata_kepuasan
    FROM '{partisi_dir}/**/*.parquet'
    WHERE tahun = 2024
      AND provinsi = 'DKI Jakarta'
    GROUP BY jenis_layanan
    ORDER BY jumlah DESC
""").df()
waktu_partisi = time.perf_counter() - start
print(f"Waktu query partisi: {waktu_partisi:.4f} detik")
print(hasil_partisi)

# %% [markdown]
# ---
# ## Bagian 6: Interoperabilitas DuckDB <-> Polars via Apache Arrow
#
# DuckDB dan Polars sama-sama berbasis Apache Arrow sebagai format in-memory.
# Pertukaran data antar keduanya hampir **zero-copy** — sangat cepat!

# %%
# Polars -> DuckDB: langsung query DataFrame Polars
print("=== Polars DataFrame langsung di-query DuckDB ===")
df_polars = pl.read_parquet(parquet_path)

# Register sebagai tabel virtual (tidak perlu copy data)
con.register("layanan_publik", df_polars)

hasil = con.execute("""
    SELECT
        provinsi,
        online,
        COUNT(*) AS jumlah,
        ROUND(AVG(durasi_menit), 1) AS rata_durasi
    FROM layanan_publik
    WHERE status = 'selesai'
    GROUP BY provinsi, online
    ORDER BY provinsi, online
""").df()
print(hasil.head(10))

# %%
# DuckDB -> Polars: konversi hasil query ke Polars DataFrame
print("\n=== Hasil DuckDB ke Polars DataFrame ===")
df_hasil_polars = con.execute("""
    SELECT
        jenis_layanan,
        SUM(biaya) AS total_pendapatan,
        COUNT(*) AS jumlah
    FROM layanan_publik
    GROUP BY jenis_layanan
""").pl()  # .pl() langsung mengonversi ke Polars DataFrame

print(type(df_hasil_polars))  # <class 'polars.dataframe.frame.DataFrame'>
print(df_hasil_polars)

# Langsung lanjutkan pipeline di Polars
df_hasil_polars = (
    df_hasil_polars
    .with_columns(
        (pl.col("total_pendapatan") / pl.col("jumlah")).alias("rata_pendapatan_per_transaksi")
    )
    .sort("total_pendapatan", descending=True)
)
print(df_hasil_polars)

# %% [markdown]
# ---
# ## Bagian 7: Perbandingan Format Data

# %%
# Benchmark: CSV vs Parquet vs Arrow (IPC)
print("=== Perbandingan Format Data ===")

import pyarrow.feather as feather

arrow_path = layanan_dir / "layanan.arrow"

# Simpan ke Arrow/Feather format
df_polars.write_ipc(arrow_path)

# Ukuran file
print("Ukuran file:")
for fmt, path in [("Parquet", parquet_path), ("CSV", csv_path), ("Arrow IPC", arrow_path)]:
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {fmt:<12}: {size_mb:.1f} MB")

# %%
# Benchmark waktu baca
print("\nWaktu membaca (Polars):")
for fmt, reader, path in [
    ("Parquet",   pl.read_parquet, parquet_path),
    ("CSV",       pl.read_csv,     csv_path),
    ("Arrow IPC", pl.read_ipc,     arrow_path),
]:
    start = time.perf_counter()
    _ = reader(path)
    durasi = time.perf_counter() - start
    print(f"  {fmt:<12}: {durasi:.4f} detik")

# %% [markdown]
# ---
# ## Bagian 8: Fitur Lanjutan DuckDB

# %%
# COPY TO: export hasil query langsung ke file
print("=== Export Hasil Query ke Parquet ===")
ringkasan_path = layanan_dir / "ringkasan.parquet"
con.execute(f"""
    COPY (
        SELECT
            provinsi,
            jenis_layanan,
            COUNT(*) AS jumlah,
            AVG(kepuasan) AS rata_kepuasan,
            SUM(biaya) AS total_pendapatan
        FROM '{parquet_path}'
        WHERE status = 'selesai'
        GROUP BY provinsi, jenis_layanan
    )
    TO '{ringkasan_path}'
    (FORMAT PARQUET)
""")
print(f"Ringkasan tersimpan ke: {ringkasan_path}")

# %%
# DuckDB: CTE (Common Table Expression) bertingkat
print("\n=== CTE Bertingkat untuk Analisis Kompleks ===")
analisis_kompleks = con.execute(f"""
    WITH
    -- CTE 1: Ringkasan per provinsi dan layanan
    ringkasan AS (
        SELECT
            provinsi,
            jenis_layanan,
            online,
            COUNT(*) AS jumlah,
            AVG(kepuasan) AS avg_kepuasan,
            AVG(durasi_menit) AS avg_durasi
        FROM '{parquet_path}'
        WHERE status = 'selesai'
        GROUP BY provinsi, jenis_layanan, online
    ),
    -- CTE 2: Hitung rata-rata nasional sebagai pembanding
    rata_nasional AS (
        SELECT
            jenis_layanan,
            AVG(avg_kepuasan) AS kepuasan_nasional
        FROM ringkasan
        GROUP BY jenis_layanan
    )
    -- Query utama: bandingkan tiap provinsi dengan rata nasional
    SELECT
        r.provinsi,
        r.jenis_layanan,
        r.online,
        r.jumlah,
        ROUND(r.avg_kepuasan, 2) AS kepuasan_lokal,
        ROUND(n.kepuasan_nasional, 2) AS kepuasan_nasional,
        ROUND(r.avg_kepuasan - n.kepuasan_nasional, 2) AS selisih
    FROM ringkasan r
    JOIN rata_nasional n USING (jenis_layanan)
    ORDER BY selisih DESC
    LIMIT 15
""").df()
print(analisis_kompleks)

# %% [markdown]
# ---
# ## Ringkasan Pertemuan 3
#
# Hari ini kita telah mempelajari:
# 1. DuckDB sebagai mesin OLAP in-process: query tanpa server, tanpa import data
# 2. Query langsung ke file Parquet, CSV, dan JSON
# 3. Predicate Pushdown dan Column Pruning: optimasi otomatis DuckDB
# 4. Window Functions SQL: RANK, ROW_NUMBER, SUM OVER, LAG, LEAD, QUALIFY
# 5. Partitioned Parquet: struktur data untuk query sangat cepat
# 6. Interoperabilitas DuckDB <-> Polars via Apache Arrow (zero-copy)
# 7. Perbandingan format: CSV (portabel), Parquet (analitik), Arrow IPC (in-memory transfer)
#
# ### Pertemuan Berikutnya
# Data Cleaning & Quality — Strategi membersihkan data kotor skala besar:
# missing values, outliers, duplikat, dan validasi otomatis dengan Great Expectations
