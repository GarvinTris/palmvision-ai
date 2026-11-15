# PalmVision AI — Deteksi Kematangan TBS Sawit (YOLO26)

Aplikasi web: unggah foto pohon sawit → model YOLO26 mendeteksi tandan dan kematangannya → pemeriksa bisa mengoreksi hasil → riwayat tersimpan di Supabase.

```
Colab (training)  →  best.onnx + model_meta.json  →  frontend/model/
                                                         │
Browser (Vercel, static) ── ONNX Runtime Web ── inferensi di perangkat pengguna
        │
        └── Supabase: Auth + Postgres (RLS) + Storage
```

Tidak ada server Python. Vercel tidak cocok untuk menjalankan PyTorch/Ultralytics (batas ukuran dan waktu fungsi), jadi inferensi dijalankan di browser dengan ONNX. Hasilnya: deploy ke Vercel cukup folder statis.

## Struktur

```
frontend/            ← ini yang di-deploy ke Vercel (Root Directory)
  index.html, app.js, yolo.js, style.css, config.js, vercel.json
  model/             ← taruh best.onnx + model_meta.json di sini
supabase/schema.sql  ← jalankan sekali di Supabase SQL Editor
training/train_yolo26_palm.ipynb  ← jalankan di Google Colab (GPU)
```

## Langkah 1 — Latih model (Colab)

1. Buka `training/train_yolo26_palm.ipynb` di Google Colab, pilih runtime GPU, jalankan semua sel.
2. Hasil akhir: `best.onnx` dan `model_meta.json` (berisi kelas, threshold per kelas, dan metrik test). File juga tersimpan di Google Drive `MyDrive/PalmVision`.
3. Salin keduanya ke `frontend/model/` (timpa placeholder `model_meta.json`).

## Langkah 2 — Supabase

1. Buat project di supabase.com.
2. SQL Editor → tempel isi `supabase/schema.sql` → Run. Ini membuat tabel `inspections`, `detections`, view `model_feedback_summary`, kebijakan RLS, dan bucket `inspections`.
3. Authentication → Providers → Email aktif. Untuk uji cepat, matikan "Confirm email" (atau biarkan aktif dan konfirmasi lewat email).
4. Project Settings → API: salin **Project URL** dan **anon public key** ke `frontend/config.js`. Jangan pakai `service_role`.

## Langkah 3 — Deploy ke Vercel

1. Push repo ke GitHub (ukuran `best.onnx` untuk yolo26s sekitar puluhan MB, aman untuk GitHub dan Vercel).
2. Vercel → Add New Project → pilih repo.
3. **Root Directory: `frontend`**, Framework Preset: *Other*, Build Command kosong, Output Directory kosong.
4. Deploy. Di Supabase → Authentication → URL Configuration, tambahkan domain Vercel ke Site URL.

Uji lokal tanpa Vercel: `cd frontend && python -m http.server 5500`, buka `http://localhost:5500`.

## Kelas dan keterbatasan data (penting)

Dataset SawitMVC melabeli B1–B4 berdasarkan **perkiraan bulan menuju panen**, bukan mentah/matang/lewat-matang. Pemetaan default notebook ini:

| SawitMVC | Kelas aplikasi |
|---|---|
| B1 (merah, besar, bulat) | `matang` |
| B2 (hitam, transisi merah) | `menjelang_matang` |
| B3 + B4 | `mentah` |

**Kelas `terlalu_matang` belum ada**, karena SawitMVC memang tidak memuatnya. Untuk menambahkannya kamu perlu data berlabel overripe (mis. dataset TBS pabrik Suharjito dkk. 2023, atau dataset Malaysia 2024), digabung ke dataset training, lalu tambahkan nama kelas di `CLASSES` pada notebook. Aplikasi membaca daftar kelas dari `model_meta.json`, jadi frontend tidak perlu diubah.

Lisensi SawitMVC adalah CC BY-NC 4.0: **hanya non-komersial**.

## Cara aplikasi menjaga hasil tetap bisa dipercaya

- **Threshold per kelas** dihitung dari data validation (F1 terbaik), bukan angka 0.25 sembarang.
- **Zona ragu-ragu**: deteksi dengan confidence dekat threshold ditandai "perlu dicek" (kotak putus-putus).
- **Koreksi manusia**: pemeriksa bisa mengganti kelas atau menolak deteksi. Setiap koreksi disimpan (`status = auto | corrected | rejected`).
- **Angka lapangan**: kartu "Deteksi diterima tanpa koreksi" menghitung persentase dari koreksi nyata. Ini ukuran yang lebih jujur daripada mAP di dataset.
- **Transparansi**: panel "Tentang model" menampilkan metrik test, threshold, dan batasan.

## Meningkatkan akurasi lebih lanjut

1. Setelah beberapa ratus koreksi terkumpul, lihat `select * from model_feedback_summary;` untuk kelas mana yang paling sering salah.
2. Ekspor foto dan kotak yang `status = 'corrected'` dari tabel `detections` + Storage, ubah ke format YOLO, dan masukkan ke training berikutnya (data dari kebunmu sendiri biasanya mengangkat akurasi paling besar).
3. Foto lebih dekat, cahaya cukup, beberapa sisi pohon. Buah tertutup pelepah tidak terdeteksi oleh model apa pun.
4. Coba `MODEL = "yolo26m.pt"` bila perangkat target cukup kuat (lebih akurat, lebih lambat di browser).

## Keamanan

- **Putar (rotate) API key Roboflow lama.** File `backend/.env` ikut terkompres di ZIP project lama. Anggap key itu sudah bocor: buat key baru di Roboflow dan hapus yang lama. Aplikasi baru tidak memakai Roboflow.
- `.env`, `*.db`, `.venv311/` sudah ada di `.gitignore`; jangan ikut di-ZIP atau di-commit (folder `.venv311` membuat ZIP lama berukuran ~440 MB).
- `anon key` Supabase memang publik; keamanan data dijaga RLS (tiap pengguna hanya mengakses datanya sendiri).
- Bucket `inspections` bersifat publik-baca (URL gambar bisa dibuka siapa pun yang tahu URL-nya). Ubah ke privat dan pakai signed URL bila foto sensitif.

## Yang belum diverifikasi

Kode training dan frontend belum dijalankan end-to-end terhadap model nyata di lingkungan penyusunnya (tanpa GPU dan tanpa akses internet). Fungsi pasca-proses (`yolo.js`) sudah diuji dengan data sintetis untuk kedua format output ONNX. Setelah training, periksa baris `output:` di sel export (YOLO26 biasanya `[1, 300, 6]`); aplikasi menangani format itu dan format klasik `[1, 4+kelas, N]`. Angka akurasi final hanya bisa diketahui setelah notebook dijalankan.
