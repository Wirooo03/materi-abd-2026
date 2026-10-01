# ---
# Pertemuan 1: Pengantar Big Data dan Ekosistem Modern
# Kuliah Analisis Big Data - Semester Ganjil 2026
# ---
#
# Notebook ini adalah panduan praktik untuk Pertemuan 1.
# Jalankan cell-cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 1: Pengantar Big Data dan Ekosistem Modern
#
# ## Tujuan Pembelajaran
# 1. Memahami konsep Big Data dan 5V
# 2. Mengenal ekosistem tools modern (DuckDB, Polars)
# 3. Membandingkan performa Pandas vs Polars
# 4. Memahami format data: CSV vs Parquet

# %% [markdown]
# ---
# ## Bagian 1: Verifikasi Environment
# Mari pastikan semua tools terinstal dengan benar.

# %%
import sys
print(f"Python version: {sys.version}")

# Core tools
from pathlib import Path

import polars as pl
import duckdb
import plotly
import plotly.express as px
import pyarrow as pa

repo_root = Path(__file__).resolve().parents[2]
slides_dir = repo_root / "slides" / "P01_pengantar"
assets_dir = slides_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
temp_dir = repo_root / ".tmp_data"
temp_dir.mkdir(parents=True, exist_ok=True)

print(f"\nPolars version:  {pl.__version__}")
print(f"DuckDB version:  {duckdb.__version__}")
print(f"Plotly version:  {plotly.__version__}")
print(f"PyArrow version: {pa.__version__}")
print(f"\nSemua tools siap digunakan.")

# %% [markdown]
# ---
# ## Bagian 2: Hello World — DuckDB dan Polars
#
# ### 2a. DuckDB: SQL langsung di Python

# %%
# DuckDB: Buat query SQL langsung di Python
result = duckdb.sql("""
    SELECT
        'Hello Big Data!' AS pesan,
        42 AS angka_ajaib,
        CURRENT_TIMESTAMP AS waktu
""")
print(result)

# %%
# DuckDB bisa generate data untuk testing
duckdb.sql("""
    SELECT
        i AS id,
        'Mahasiswa_' || i AS nama,
        CASE WHEN i % 3 = 0 THEN 'Teknik Informatika'
             WHEN i % 3 = 1 THEN 'Sistem Informasi'
             ELSE 'Data Science'
        END AS jurusan,
        60 + (random() * 40)::INT AS nilai
    FROM generate_series(1, 10) AS t(i)
""").show()

# %% [markdown]
# ### 2b. Polars: DataFrame Modern

# %%
# Polars: Buat DataFrame
df = pl.DataFrame(
    {
        "nama": ["Andi", "Budi", "Citra", "Dewi", "Eka"],
        "kota": ["Jakarta", "Bandung", "Surabaya", "Jakarta", "Bandung"],
        "nilai": [85, 92, 78, 88, 95],
        "semester": [5, 5, 5, 7, 5],
    }
)
print(df)

# %%
# Polars expressions: filter, group, aggregate
hasil = (
    df
    .filter(pl.col("semester") == 5)
    .group_by("kota")
    .agg(
        pl.col("nilai").mean().alias("rata_rata_nilai"),
        pl.col("nama").count().alias("jumlah_mahasiswa"),
    )
    .sort("rata_rata_nilai", descending=True)
)
print(hasil)

# %% [markdown]
# ### 2c. Interoperabilitas DuckDB dan Polars

# %%
# DuckDB bisa query Polars DataFrame langsung
duckdb.sql("""
    SELECT kota, AVG(nilai) AS avg_nilai
    FROM df
    WHERE semester = 5
    GROUP BY kota
    ORDER BY avg_nilai DESC
""").show()

# %%
# Sebaliknya: DuckDB result ke Polars DataFrame
result_polars = duckdb.sql("""
    SELECT * FROM df WHERE nilai > 85
""").pl()

print(type(result_polars))
print(result_polars)

# %% [markdown]
# ---
# ## Bagian 3: Benchmark — Pandas vs Polars
#
# Mari kita buat dataset besar dan bandingkan performanya.

# %%
import time
import numpy as np
import pandas as pd

# Generate dataset 1 juta baris
N = 1_000_000
print(f"Generating {N:,} baris data...")

# Data untuk Pandas dan Polars
np.random.seed(42)
data = {
    "id": range(N),
    "kota": np.random.choice(
        ["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang",
         "Makassar", "Palembang", "Yogyakarta", "Denpasar", "Manado"],
        N,
    ),
    "kategori": np.random.choice(
        ["Elektronik", "Fashion", "Makanan", "Kesehatan", "Olahraga"],
        N,
    ),
    "harga": np.random.uniform(10_000, 5_000_000, N).astype(int),
    "kuantitas": np.random.randint(1, 50, N),
}

# %% [markdown]
# ### 3a. Pandas

# %%
# Pandas
start = time.perf_counter()
df_pandas = pd.DataFrame(data)
result_pandas = (
    df_pandas
    .groupby(["kota", "kategori"])
    .agg(
        total_penjualan=("harga", "sum"),
        rata_rata_harga=("harga", "mean"),
        jumlah_transaksi=("id", "count"),
    )
    .reset_index()
    .sort_values("total_penjualan", ascending=False)
    .head(10)
)
waktu_pandas = time.perf_counter() - start
print(f"Pandas: {waktu_pandas:.4f} detik")
print(result_pandas)

# %% [markdown]
# ### 3b. Polars

# %%
# Polars
start = time.perf_counter()
df_polars = pl.DataFrame(data)
result_polars = (
    df_polars
    .group_by(["kota", "kategori"])
    .agg(
        pl.col("harga").sum().alias("total_penjualan"),
        pl.col("harga").mean().alias("rata_rata_harga"),
        pl.col("id").count().alias("jumlah_transaksi"),
    )
    .sort("total_penjualan", descending=True)
    .head(10)
)
waktu_polars = time.perf_counter() - start
print(f"Polars: {waktu_polars:.4f} detik")
print(result_polars)

# %%
# Perbandingan
speedup = waktu_pandas / waktu_polars
print(f"\nHasil Benchmark ({N:,} baris):")
print(f"   Pandas: {waktu_pandas:.4f} detik")
print(f"   Polars: {waktu_polars:.4f} detik")
print(f"   Speedup: {speedup:.1f}x lebih cepat!")

# %% [markdown]
# ### 3c. Grafik Benchmark
#
# Simpan grafik ke folder assets agar dapat digunakan di slide Marp.

# %%
import plotly.graph_objects as go
import matplotlib.pyplot as plt

labels = ["Pandas", "Polars"]
waktu = [waktu_pandas, waktu_polars]

fig = go.Figure(
    go.Bar(
        x=labels,
        y=waktu,
        marker_color=["#e74c3c", "#2ecc71"],
        text=[f"{nilai:.4f} s" for nilai in waktu],
        textposition="auto",
    )
)
fig.update_layout(
    title=f"Benchmark Pandas vs Polars ({N:,} Baris)",
    xaxis_title="Library",
    yaxis_title="Waktu (detik)",
    template="plotly_white",
)
fig.show()
plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, waktu, color=["#e74c3c", "#2ecc71"])
plt.title(f"Benchmark Pandas vs Polars ({N:,} Baris)")
plt.ylabel("Waktu (detik)")
plt.grid(axis="y", alpha=0.25)
for bar, nilai in zip(bars, waktu):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{nilai:.4f} s", ha="center", va="bottom")
plt.tight_layout()
output_png = assets_dir / "benchmark.png"
plt.savefig(output_png, dpi=160)
plt.close()
print(f"Grafik disimpan ke {output_png.relative_to(repo_root)}")

# %% [markdown]
# ---
# ## Bagian 4: Format Data — CSV vs Parquet
#
# Parquet adalah format data kolumnar yang jauh lebih efisien daripada CSV.

# %%
import os

# Simpan sebagai CSV dan Parquet
csv_path = temp_dir / "demo_data.csv"
parquet_path = temp_dir / "demo_data.parquet"

df_polars.write_csv(csv_path)
df_polars.write_parquet(parquet_path)

csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB

print(f"Ukuran file ({N:,} baris):")
print(f"   CSV:     {csv_size:.2f} MB")
print(f"   Parquet: {parquet_size:.2f} MB")
print(f"   Rasio:   Parquet {csv_size / parquet_size:.1f}x lebih kecil!")

# %%
# Benchmark: membaca CSV vs Parquet
start = time.perf_counter()
_ = pl.read_csv(csv_path)
waktu_csv = time.perf_counter() - start

start = time.perf_counter()
_ = pl.read_parquet(parquet_path)
waktu_parquet = time.perf_counter() - start

print(f"\nWaktu baca:")
print(f"   CSV:     {waktu_csv:.4f} detik")
print(f"   Parquet: {waktu_parquet:.4f} detik")
print(f"   Speedup: {waktu_csv / waktu_parquet:.1f}x lebih cepat!")

# %%
# DuckDB: Query langsung ke file Parquet (tanpa load ke memory)
duckdb.sql(f"""
    SELECT kota, COUNT(*) AS jumlah, AVG(harga) AS avg_harga
    FROM '{parquet_path}'
    GROUP BY kota
    ORDER BY jumlah DESC
""").show()

# %% [markdown]
# ---
# ## Bagian 5: Visualisasi Pertama dengan Plotly

# %%
# Aggregate data untuk visualisasi
viz_data = (
    df_polars
    .group_by("kota")
    .agg(
        pl.col("harga").sum().alias("total_penjualan"),
        pl.col("id").count().alias("jumlah_transaksi"),
    )
    .sort("total_penjualan", descending=True)
)

# Plotly bar chart
fig = px.bar(
    viz_data.to_pandas(),
    x="kota",
    y="total_penjualan",
    color="jumlah_transaksi",
    title="Total Penjualan per Kota (Data Simulasi)",
    labels={
        "kota": "Kota",
        "total_penjualan": "Total Penjualan (Rp)",
        "jumlah_transaksi": "Jumlah Transaksi",
    },
    color_continuous_scale="Viridis",
)
fig.update_layout(
    xaxis_tickangle=-45,
    template="plotly_dark",
)
fig.show()

# %%
# Scatter plot: harga vs kuantitas (sample 10,000 dari 1 juta)
sample = df_polars.sample(n=10_000, seed=42)
fig2 = px.scatter(
    sample.to_pandas(),
    x="harga",
    y="kuantitas",
    color="kategori",
    title="Harga vs Kuantitas per Kategori (Sample 10K dari 1M)",
    labels={"harga": "Harga (Rp)", "kuantitas": "Kuantitas"},
    opacity=0.5,
    template="plotly_dark",
)
fig2.show()

# %% [markdown]
# ---
# ## Ringkasan Pertemuan 1
#
# Hari ini kita telah:
# 1. Setup environment (Docker + JupyterLab)
# 2. Berkenalan dengan DuckDB (SQL analytics) dan Polars (DataFrame modern)
# 3. Melihat interoperabilitas DuckDB dan Polars via Apache Arrow
# 4. Benchmark: Polars jauh lebih cepat dari Pandas
# 5. Format data: Parquet jauh lebih efisien dari CSV
# 6. Visualisasi pertama dengan Plotly
#
# ### Pertemuan Berikutnya
# Polars Deep Dive — Lazy evaluation, expressions, window functions
