# ---
# Pertemuan 2: Python untuk Big Data — Polars Deep Dive
# Kuliah Analisis Big Data - Semester Ganjil 2026
# ---
#
# Notebook ini adalah panduan praktik untuk Pertemuan 2.
# Jalankan cell-cell di bawah ini secara berurutan.

# %% [markdown]
# # Pertemuan 2: Python untuk Big Data — Polars Deep Dive
#
# ## Tujuan Pembelajaran
# 1. Memahami arsitektur Polars: Rust, Apache Arrow, dan eksekusi multi-thread
# 2. Membedakan dan memilih antara Lazy vs Eager API
# 3. Menguasai Polars Expressions untuk manipulasi data kolumnar
# 4. Mengoptimalkan pipeline dengan chaining, predicate pushdown, dan projection pushdown
# 5. Menulis data ke format Parquet yang efisien

# %% [markdown]
# ---
# ## Bagian 1: Setup dan Pembuatan Data Simulasi
#
# Kita akan membuat dataset e-commerce simulasi Indonesia dengan 2 juta baris
# untuk mendemonstrasikan keunggulan Polars pada data berskala besar.

# %%
from pathlib import Path

import polars as pl
import numpy as np
import time
import os

repo_root = Path(__file__).resolve().parents[2]
temp_dir = repo_root / ".tmp_data"
temp_dir.mkdir(parents=True, exist_ok=True)

# Seed untuk reproduktibilitas
np.random.seed(2026)

N = 2_000_000
print(f"Membuat dataset simulasi {N:,} baris...")

kota_list = [
    "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang",
    "Makassar", "Palembang", "Tangerang", "Depok", "Bogor",
    "Yogyakarta", "Bekasi", "Denpasar", "Balikpapan", "Manado",
]
kategori_list = ["Elektronik", "Fashion", "Makanan", "Kesehatan", "Olahraga", "Kecantikan", "Otomotif"]
status_list = ["selesai", "dikirim", "proses", "dibatalkan", "dikembalikan"]

data = {
    "transaksi_id": [f"TRX{i:08d}" for i in range(N)],
    "tanggal": pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 12, 31), interval="1s",
        eager=True
    ).sample(N, with_replacement=True, seed=2026).alias("tanggal"),
    "kota_penjual": np.random.choice(kota_list, N),
    "kota_pembeli": np.random.choice(kota_list, N),
    "kategori": np.random.choice(kategori_list, N),
    "harga": np.random.randint(5_000, 10_000_000, N),
    "kuantitas": np.random.randint(1, 20, N),
    "diskon_persen": np.random.choice([0, 0, 0, 5, 10, 15, 20, 25, 30, 50], N),
    "rating": np.round(np.random.uniform(1.0, 5.0, N), 1),
    "status": np.random.choice(status_list, N, p=[0.65, 0.15, 0.10, 0.06, 0.04]),
}

# Buat DataFrame Polars
df = pl.DataFrame(data)
print(f"Dataset berhasil dibuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
print(df.head(5))

# %% [markdown]
# ---
# ## Bagian 2: Eager vs Lazy API
#
# Polars memiliki dua mode eksekusi:
# - **Eager**: Langsung dieksekusi saat perintah ditulis (seperti Pandas)
# - **Lazy**: Membangun rencana eksekusi dulu, baru dijalankan saat `.collect()` dipanggil
#
# Mode Lazy memungkinkan Polars mengoptimalkan query secara otomatis.

# %%
# --- EAGER API ---
# Eksekusi langsung, hasilnya langsung tersedia
print("=== Eager API ===")
start = time.perf_counter()
hasil_eager = (
    df
    .filter(pl.col("status") == "selesai")
    .group_by("kategori")
    .agg(
        pl.col("harga").mean().alias("rata_rata_harga"),
        pl.col("kuantitas").sum().alias("total_kuantitas"),
    )
    .sort("rata_rata_harga", descending=True)
)
waktu_eager = time.perf_counter() - start
print(f"Waktu Eager: {waktu_eager:.4f} detik")
print(hasil_eager)

# %%
# --- LAZY API ---
# Membangun rencana eksekusi (LazyFrame), baru dieksekusi saat .collect()
print("\n=== Lazy API ===")
start = time.perf_counter()
hasil_lazy = (
    df.lazy()                                     # Ubah ke LazyFrame
    .filter(pl.col("status") == "selesai")        # Filter dulu (predicate pushdown)
    .group_by("kategori")
    .agg(
        pl.col("harga").mean().alias("rata_rata_harga"),
        pl.col("kuantitas").sum().alias("total_kuantitas"),
    )
    .sort("rata_rata_harga", descending=True)
    .collect()                                    # Baru dieksekusi di sini
)
waktu_lazy = time.perf_counter() - start
print(f"Waktu Lazy:  {waktu_lazy:.4f} detik")
print(hasil_lazy)

# %%
# Melihat rencana eksekusi Lazy (query plan)
print("\n=== Query Plan (Lazy) ===")
q = (
    df.lazy()
    .filter(pl.col("status") == "selesai")
    .group_by("kategori")
    .agg(pl.col("harga").mean().alias("rata_rata_harga"))
    .sort("rata_rata_harga", descending=True)
)
# Tampilkan rencana query yang dioptimalkan
print(q.explain(optimized=True))

# %% [markdown]
# ---
# ## Bagian 3: Polars Expressions
#
# Expressions adalah inti dari Polars. Berbeda dengan Pandas yang bekerja baris per baris,
# Polars bekerja kolom per kolom secara paralel menggunakan Apache Arrow.
#
# ### 3a. Ekspresi Dasar: select dan with_columns

# %%
# select: memilih dan mentransformasi kolom
print("=== select: memilih kolom ===")
hasil = df.select([
    pl.col("transaksi_id"),
    pl.col("kategori"),
    pl.col("harga"),
    pl.col("kuantitas"),
    # Membuat kolom baru: total = harga * kuantitas * (1 - diskon/100)
    (pl.col("harga") * pl.col("kuantitas") * (1 - pl.col("diskon_persen") / 100))
    .alias("total_bayar"),
])
print(hasil.head(5))

# %%
# with_columns: menambahkan kolom baru tanpa menghapus kolom lama
print("\n=== with_columns: menambahkan kolom ===")
df_enriched = df.with_columns([
    # Total bayar setelah diskon
    (pl.col("harga") * pl.col("kuantitas") * (1 - pl.col("diskon_persen") / 100))
    .cast(pl.Int64)
    .alias("total_bayar"),
    # Klasifikasi harga
    pl.when(pl.col("harga") < 100_000).then(pl.lit("Murah"))
    .when(pl.col("harga") < 1_000_000).then(pl.lit("Menengah"))
    .otherwise(pl.lit("Premium"))
    .alias("segmen_harga"),
    # Apakah transaksi berhasil?
    (pl.col("status") == "selesai").alias("berhasil"),
])
print(df_enriched.select(["transaksi_id", "harga", "total_bayar", "segmen_harga", "berhasil"]).head(8))

# %% [markdown]
# ### 3b. Filter: Kondisi dan Logika Boolean

# %%
# Filter dengan berbagai kondisi
print("=== Filter multi-kondisi ===")

# Transaksi besar yang selesai di Jakarta atau Surabaya
transaksi_besar = (
    df
    .filter(
        (pl.col("status") == "selesai") &
        (pl.col("harga") > 1_000_000) &
        (pl.col("kota_penjual").is_in(["Jakarta", "Surabaya"]))
    )
)
print(f"Jumlah transaksi besar: {len(transaksi_besar):,}")
print(transaksi_besar.head(5))

# %%
# Filter dengan ekspresi string
print("\n=== Filter dengan string operations ===")
transaksi_elektronik = df.filter(
    pl.col("kategori").str.to_lowercase().str.contains("elekt")
)
print(f"Transaksi elektronik: {len(transaksi_elektronik):,}")

# %% [markdown]
# ### 3c. group_by dan Agregasi

# %%
# Agregasi lengkap per kategori
print("=== Agregasi per kategori ===")
ringkasan_kategori = (
    df_enriched
    .group_by("kategori")
    .agg([
        pl.col("transaksi_id").count().alias("jumlah_transaksi"),
        pl.col("total_bayar").sum().alias("total_pendapatan"),
        pl.col("total_bayar").mean().alias("rata_rata_order"),
        pl.col("rating").mean().round(2).alias("rata_rata_rating"),
        pl.col("berhasil").sum().alias("transaksi_selesai"),
        pl.col("diskon_persen").mean().round(1).alias("rata_rata_diskon"),
    ])
    .with_columns(
        (pl.col("transaksi_selesai") / pl.col("jumlah_transaksi") * 100)
        .round(1)
        .alias("persen_berhasil")
    )
    .sort("total_pendapatan", descending=True)
)
print(ringkasan_kategori)

# %%
# Agregasi per kota dan kategori (multi-key group_by)
print("\n=== Top 5 kombinasi kota-kategori ===")
top_kombinasi = (
    df_enriched
    .filter(pl.col("status") == "selesai")
    .group_by(["kota_penjual", "kategori"])
    .agg(
        pl.col("total_bayar").sum().alias("total_pendapatan"),
        pl.col("transaksi_id").count().alias("jumlah"),
    )
    .sort("total_pendapatan", descending=True)
    .head(10)
)
print(top_kombinasi)

# %% [markdown]
# ---
# ## Bagian 4: Lazy Loading dengan scan_parquet dan scan_csv
#
# `scan_csv` dan `scan_parquet` adalah versi lazy dari `read_csv`/`read_parquet`.
# Data belum dibaca ke memori sampai `.collect()` dipanggil.
# Ini sangat efisien untuk file berukuran sangat besar.

# %%
# Simpan dataset ke file terlebih dahulu
print("Menyimpan data ke file...")
bigdata_dir = temp_dir / "bigdata_demo"
bigdata_dir.mkdir(parents=True, exist_ok=True)
parquet_path = bigdata_dir / "transaksi.parquet"
csv_path = bigdata_dir / "transaksi.csv"

df.write_parquet(parquet_path)
df.write_csv(csv_path)
print(f"File tersimpan: {parquet_path}")
print(f"  Parquet: {os.path.getsize(parquet_path)/1e6:.1f} MB")
print(f"  CSV:     {os.path.getsize(csv_path)/1e6:.1f} MB")

# %%
# scan_parquet: lazy loading — hanya membaca kolom yang dibutuhkan
print("\n=== scan_parquet (Lazy) ===")
start = time.perf_counter()

hasil_scan = (
    pl.scan_parquet(parquet_path)      # Belum membaca file
    .filter(pl.col("status") == "selesai")   # Filter akan di-pushdown ke pembaca file
    .select(["kategori", "harga", "kuantitas"])  # Hanya baca 3 kolom (projection pushdown)
    .group_by("kategori")
    .agg(pl.col("harga").sum().alias("total"))
    .sort("total", descending=True)
    .collect()                                   # Baru baca file di sini
)
waktu_scan = time.perf_counter() - start
print(f"Waktu scan_parquet: {waktu_scan:.4f} detik")
print(hasil_scan)

# %%
# Perbandingan: read_parquet (eager) vs scan_parquet (lazy)
print("\n=== Perbandingan read_parquet vs scan_parquet ===")

# Eager: baca semua ke memori dulu
start = time.perf_counter()
_ = (
    pl.read_parquet(parquet_path)
    .filter(pl.col("status") == "selesai")
    .group_by("kategori")
    .agg(pl.col("harga").sum())
)
waktu_eager = time.perf_counter() - start

# Lazy: hanya baca kolom dan baris yang diperlukan
start = time.perf_counter()
_ = (
    pl.scan_parquet(parquet_path)
    .filter(pl.col("status") == "selesai")
    .select(["kategori", "harga"])
    .group_by("kategori")
    .agg(pl.col("harga").sum())
    .collect()
)
waktu_lazy = time.perf_counter() - start

print(f"read_parquet (eager): {waktu_eager:.4f} detik")
print(f"scan_parquet (lazy):  {waktu_lazy:.4f} detik")
print(f"Speedup lazy:         {waktu_eager/waktu_lazy:.1f}x lebih cepat")

# %% [markdown]
# ---
# ## Bagian 5: Joins di Polars

# %%
# Buat tabel dimensi (master data kategori)
kategori_info = pl.DataFrame({
    "kategori": kategori_list,
    "divisi": ["Teknologi", "Gaya Hidup", "Konsumsi", "Kesehatan", "Gaya Hidup", "Kecantikan", "Otomotif"],
    "margin_persen": [12, 35, 8, 25, 30, 40, 15],
    "target_bulanan": [50_000_000_000, 30_000_000_000, 20_000_000_000,
                       15_000_000_000, 10_000_000_000, 8_000_000_000, 25_000_000_000],
})
print("=== Tabel Dimensi Kategori ===")
print(kategori_info)

# %%
# Inner join: gabungkan transaksi dengan info kategori
print("\n=== Inner Join: Transaksi + Info Kategori ===")
df_joined = (
    df_enriched
    .join(kategori_info, on="kategori", how="inner")
    .with_columns(
        (pl.col("total_bayar") * pl.col("margin_persen") / 100)
        .alias("estimasi_profit")
    )
)
print(f"Jumlah baris setelah join: {len(df_joined):,}")
print(df_joined.select(["transaksi_id", "kategori", "divisi", "total_bayar", "estimasi_profit"]).head(5))

# %%
# Analisis per divisi setelah join
print("\n=== Analisis per Divisi ===")
analisis_divisi = (
    df_joined
    .filter(pl.col("status") == "selesai")
    .group_by("divisi")
    .agg([
        pl.col("total_bayar").sum().alias("total_pendapatan"),
        pl.col("estimasi_profit").sum().alias("total_profit"),
        pl.col("transaksi_id").count().alias("jumlah_transaksi"),
    ])
    .sort("total_pendapatan", descending=True)
)
print(analisis_divisi)

# %% [markdown]
# ---
# ## Bagian 6: Window Functions
#
# Window functions memungkinkan agregasi dalam grup tanpa mereduksi jumlah baris.
# Berguna untuk: ranking, running total, perbandingan dengan rata-rata grup, dll.

# %%
# Agregasi per kota untuk window function
print("=== Window Functions ===")

df_kota = (
    df_enriched
    .filter(pl.col("status") == "selesai")
    .group_by(["kota_penjual", "kategori"])
    .agg(pl.col("total_bayar").sum().alias("pendapatan"))
)

df_window = df_kota.with_columns([
    # Ranking pendapatan per kota (dalam grup kota_penjual)
    pl.col("pendapatan")
    .rank(method="dense", descending=True)
    .over("kota_penjual")
    .alias("rank_dalam_kota"),

    # Rata-rata pendapatan per kota (window aggregation)
    pl.col("pendapatan")
    .mean()
    .over("kota_penjual")
    .alias("rata_rata_kota"),

    # Persentase kontribusi terhadap total kota
    (pl.col("pendapatan") / pl.col("pendapatan").sum().over("kota_penjual") * 100)
    .round(1)
    .alias("persen_kontribusi"),
])

# Tampilkan top kategori per kota Jakarta
print(df_window.filter(pl.col("kota_penjual") == "Jakarta").sort("rank_dalam_kota"))

# %% [markdown]
# ---
# ## Bagian 7: String Operations dan Temporal

# %%
# Operasi string dengan Polars
print("=== String Operations ===")
df_string = df.select([
    pl.col("transaksi_id"),
    pl.col("kota_penjual"),
    # Extract nomor dari ID transaksi
    pl.col("transaksi_id").str.replace("TRX", "").cast(pl.Int32).alias("nomor_urut"),
    # Gabungkan kota penjual dan pembeli
    (pl.col("kota_penjual") + " -> " + pl.col("kota_pembeli")).alias("rute"),
    # Apakah pengiriman dalam kota yang sama?
    (pl.col("kota_penjual") == pl.col("kota_pembeli")).alias("pengiriman_lokal"),
])
print(df_string.head(8))

# %%
# Operasi temporal (date/datetime)
print("\n=== Temporal Operations ===")
df_temporal = df.with_columns([
    pl.col("tanggal").dt.month().alias("bulan"),
    pl.col("tanggal").dt.quarter().alias("kuartal"),
    pl.col("tanggal").dt.weekday().alias("hari_dalam_minggu"),  # 0=Senin, 6=Minggu
]).select([
    "tanggal", "bulan", "kuartal", "hari_dalam_minggu",
    "kategori", "harga"
])

# Analisis penjualan per bulan
print("Penjualan per bulan:")
print(
    df_temporal
    .group_by("bulan")
    .agg(pl.col("harga").sum().alias("total_penjualan"))
    .sort("bulan")
)

# %% [markdown]
# ---
# ## Bagian 8: Menulis Data ke Parquet dengan Partisi
#
# Partisi Parquet memungkinkan query yang jauh lebih cepat karena hanya
# folder/partisi yang relevan yang dibaca.

# %%
# Simpan dengan partisi per kategori
print("=== Menulis Parquet dengan Partisi ===")
output_dir = bigdata_dir / "transaksi_partisi"

(
    df_enriched
    .with_columns(pl.col("tanggal").dt.month().alias("bulan"))
    .write_parquet(
        output_dir,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": ["kategori", "bulan"]},
    )
)
print(f"Data terpartisi tersimpan di: {output_dir}")

# Lihat struktur folder partisi
for dirpath, dirnames, filenames in os.walk(output_dir):
    depth = os.path.relpath(dirpath, output_dir).count(os.sep)
    if depth <= 2:
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")
        if filenames and depth == 2:
            for f in filenames[:1]:
                print(f"{indent}  {f}")

# %%
# Query data terpartisi: hanya baca partisi Elektronik
print("\n=== Query pada Data Terpartisi ===")
start = time.perf_counter()
hasil_partisi = (
    pl.scan_parquet(f"{output_dir}/kategori=Elektronik/**/*.parquet")
    .filter(pl.col("status") == "selesai")
    .select([
        pl.col("total_bayar").sum().alias("total"),
        pl.col("transaksi_id").count().alias("jumlah"),
    ])
    .collect()
)
print(f"Waktu query: {time.perf_counter()-start:.4f} detik")
print(hasil_partisi)

# %% [markdown]
# ---
# ## Ringkasan Pertemuan 2
#
# Hari ini kita telah mempelajari:
# 1. Arsitektur Polars: Rust + Apache Arrow = kecepatan dan efisiensi memori
# 2. Lazy vs Eager API: gunakan Lazy untuk data besar agar query dioptimalkan otomatis
# 3. Polars Expressions: cara bekerja kolumnar yang efisien
# 4. scan_parquet / scan_csv: lazy loading untuk file berukuran besar
# 5. Joins: inner, left, outer join di Polars
# 6. Window Functions: agregasi dalam grup tanpa mengurangi baris
# 7. String & Temporal operations
# 8. Partisi Parquet untuk query yang lebih cepat
#
# ### Pertemuan Berikutnya
# DuckDB & Format Data — SQL analitik in-process, query langsung ke file Parquet,
# dan interoperabilitas DuckDB <-> Polars via Apache Arrow
