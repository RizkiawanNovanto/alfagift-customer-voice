# Alfagift Customer Voice Analytics

Dashboard internal untuk membaca suara pelanggan dari ulasan aplikasi Alfagift.
Setiap ulasan diberi sentimen oleh **IndoBERT** hasil fine-tuning (model wajib, tanpa fallback).

## 1. Struktur folder

```
app.py                  ← file utama Streamlit
requirements.txt        ← daftar paket
models/indobert/        ← taruh model di sini (lihat bagian 2)
data/final/             ← data review yang ditampilkan dashboard
src/                    ← logika prediksi, data, dan tampilan halaman
assets/                 ← logo & favicon
```

## 2. Letakkan model IndoBERT

Model hasil training ada di komputer Anda sebagai `indobert/final/`. Yang perlu dipindah:

| File | Lokasi di proyek ini | Sudah ada? |
|---|---|---|
| `model.safetensors` (±450 MB) | `models/indobert/model.safetensors` | **Belum, salin dari `indobert/final/`** |
| `config.json` | `models/indobert/config.json` | Sudah |
| `tokenizer.json` | `models/indobert/tokenizer.json` | Sudah |
| `tokenizer_config.json` | `models/indobert/tokenizer_config.json` | Sudah |
| `training_args.bin` | `models/indobert/training_args.bin` | Sudah (opsional) |

Salin `model.safetensors` ke `models/indobert/`. Jangan dimasukkan ke GitHub: file ini melebihi batas 100 MB per file, dan `.gitignore` sudah memblokirnya.

## 3. Jalankan di komputer sendiri (Windows)

Buka PowerShell di folder proyek, lalu jalankan:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`.

## 4. Deploy: langkah demi langkah

Ada tiga tempat yang dibuka: **Hugging Face** (tempat model), **GitHub** (kode), dan **Streamlit Community Cloud** (tempat dashboard berjalan).

### a. Upload model ke Hugging Face
1. Buka huggingface.co, daftar atau login, lalu klik **New → Model**.
2. Beri nama, misalnya `alfagift-indobert`. Pilih *Private* bila tidak ingin publik. Klik **Create**.
3. Buka tab **Files and versions → Add file → Upload files**. Unggah `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, dan `training_args.bin`.
4. Catat ID modelnya, misalnya `username/alfagift-indobert`.
5. Jika repo *Private*: buat token di **Settings → Access Tokens** (tipe *Read*), lalu simpan token itu untuk langkah c.

### b. Upload kode ke GitHub
Di folder proyek:

```bash
git init
git add .
git commit -m "Dashboard Alfagift Customer Voice"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Buat repo kosong di GitHub dulu (tanpa README), lalu ganti `USERNAME/NAMA-REPO`. `model.safetensors` otomatis tidak ikut terunggah.

### c. Deploy ke Streamlit Community Cloud
1. Buka share.streamlit.io, login dengan akun GitHub, lalu klik **Create app → Deploy a public app from GitHub**.
2. Isi **Repository** (repo dari langkah b), **Branch** `main`, dan **Main file path** `app.py`.
3. Klik **Advanced settings → Secrets**, lalu isi:
   ```toml
   ALFAGIFT_MODEL_DIR = "username/alfagift-indobert"
   HF_TOKEN = "hf_xxxxxxxx"   # hanya jika repo Hugging Face-nya Private
   ```
4. Klik **Deploy**. Pemasangan pertama memakan beberapa menit. Model diunduh dari Hugging Face saat aplikasi pertama kali dibuka.

## 5. Penggunaan dashboard

- **Ringkasan, Sentimen, Masalah Pengguna, Wawasan, Tren, Kata Kunci, Versi Aplikasi, Review**: membaca data review yang sudah diprediksi IndoBERT.
- **Coba Prediksi**: tulis ulasan baru, beri bintang, lalu IndoBERT membaca sentimennya secara langsung.
- **Performa Model**: akurasi, matriks kebingungan, dan review yang salah diprediksi.
- **Pengaturan lanjutan** (sidebar): *Hitung ulang dengan IndoBERT*. Secara default dashboard memakai hasil prediksi yang sudah tersimpan (cepat). Aktifkan opsi ini bila ingin IndoBERT dijalankan ulang pada setiap review. Proses ini memakan ±10 menit di server kecil.

## 6. Pemecahan masalah

| Pesan / gejala | Penyebab | Solusi |
|---|---|---|
| "Model IndoBERT belum ditemukan" | `model.safetensors` belum ada di `models/indobert/` | Salin file ke folder itu (lokal), atau isi `ALFAGIFT_MODEL_DIR` (deploy) |
| Build gagal karena ukuran / timeout | Paket torch terlalu besar | Hapus baris pertama `requirements.txt` (`--extra-index-url ...`), lalu deploy ulang |
| Aplikasi "resource limit" / restart terus | RAM server gratis terbatas untuk model ±450 MB | Pindah ke Hugging Face Spaces (SDK Streamlit), kodenya tidak perlu diubah |
| Halaman lambat pertama kali | Model diunduh atau dimuat ulang | Tunggu sebentar; kunjungan berikutnya lebih cepat |

## 7. Catatan

- Label sentimen pada data latih berasal dari rating bintang, sehingga kasus bintang tinggi dengan teks mengeluh tetap ditandai di dashboard.
- Dashboard ini memakai data review dari periode yang tersimpan di `data/final/`. Untuk data baru, ekspor ulasan lalu jalankan prediksi IndoBERT dari notebook.
