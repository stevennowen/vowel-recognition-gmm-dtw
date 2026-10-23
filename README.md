# vowel-recognition-gmm-dtw

Pengenalan 5 vowel Bahasa Indonesia (/a/, /i/, /u/, /e/, /o/) dengan dua sistem ASR:

1. **Template-based ASR dengan Dynamic Time Warping (DTW)** memakai averaged template.
2. **ASR berbasis Model Gaussian** dengan mean dan covariance, dipilih berdasarkan negative log-likelihood terkecil.

Praktikum IF4071 Pemrosesan Ucapan, Minggu ke-9.

## Struktur Repo

```
vowel-recognition-gmm-dtw/
├── config.py             parameter audio, MFCC, model, dan path
├── main.py               menjalankan seluruh eksperimen
├── requirements.txt
├── src/
│   ├── data_loader.py    preprocessing dan data loading
│   ├── features.py       ekstraksi MFCC 39 dimensi
│   ├── dtw.py            DTW dan pembentukan averaged template
│   ├── gaussian.py       pembentukan model Gaussian dan NLL
│   ├── recognizer.py     recognition untuk kedua sistem
│   └── evaluate.py       perhitungan akurasi dan tabel hasil
├── data/
│   ├── template/         data latih (template)
│   └── test/
│       ├── closed/       data uji closed speakers
│       └── open/         data uji open speakers
└── results/              hasil eksperimen
```

## Dataset

Format file adalah WAV dengan sample rate berapa pun, mono atau stereo. Semua audio otomatis diubah ke 16 kHz mono.

Nama file mengikuti format `<speaker>_<vowel>.wav`, misalnya `s1_a.wav` atau `budi_o.wav`.

| Folder | Isi | Jumlah |
|---|---|---|
| `data/template/` | Rekaman pertama dari 5 anggota kelompok | 5 speaker x 5 vowel = 25 |
| `data/test/closed/` | Rekaman kedua dari 5 anggota yang sama | 5 speaker x 5 vowel = 25 |
| `data/test/open/` | Rekaman dari anggota kelompok lain | n speaker x 5 vowel |

ID speaker di `template/` dan `test/closed/` harus sama agar tabel per speaker bisa dibentuk.

Skenario pengujian:

| Skenario | Data uji |
|---|---|
| Closed Recordings | `data/template/` (rekaman yang sama dengan yang dipakai membentuk template) |
| Closed Speakers | `data/test/closed/` (speaker sama, rekaman berbeda) |
| Open Speakers | `data/test/open/` (speaker dari kelompok lain) |

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan

```bash
python main.py
```

Ringkasan akurasi dicetak di terminal. Hasil lengkap tersimpan di `results/`:

- `results.md` berisi semua tabel dalam format Markdown
- `summary.csv` untuk Tabel 1, akurasi per sistem dan skenario beserta rata-rata
- `per_vowel.csv` untuk Tabel 2, akurasi per vowel
- `closed_speakers.csv` untuk Tabel 3, closed experiments per speaker
- `open_speakers.csv` untuk Tabel 4, open experiments per speaker
- `confusion_*.csv` berisi confusion matrix setiap sistem dan skenario, untuk analisis kesalahan vowel
- `predictions.csv` berisi prediksi setiap file

Skenario yang foldernya masih kosong akan dilewati.

## Metode

### Preprocessing

1. Audio dibaca, diubah ke mono, di-resample ke 16 kHz, lalu dinormalisasi ke amplitudo puncak 1.
2. Hening di awal dan akhir dipotong. Frame 25 ms yang energinya lebih dari 30 dB di bawah frame terkeras dianggap hening. Langkah ini penting untuk Model Gaussian, karena frame hening akan ikut membentuk mean dan covariance.

### Ekstraksi Fitur MFCC 39 Dimensi

Menggunakan `python_speech_features`:

| Parameter | Nilai |
|---|---|
| Panjang frame / pergeseran | 25 ms / 10 ms |
| Preemphasis | 0.97 |
| Fungsi jendela | Hamming |
| NFFT | 512 |
| Jumlah mel filter | 23 |
| Koefisien statis | 13 (log energi frame + cepstral 1 sampai 12) |
| Delta dan delta-delta | D = 2 |

Vektor per frame: 13 statis + 13 delta + 13 delta-delta = **39 dimensi**.

CMVN per ucapan sengaja tidak dipakai. Untuk ucapan berupa satu vowel, rata-rata cepstrum justru membawa ciri vowel tersebut, sehingga mengurangkannya akan menghapus informasi yang membedakan antar vowel.

### Sistem 1: DTW dengan Averaged Template

Biaya lokal adalah jarak Euclidean antara frame template ke-i dan frame input ke-j. Biaya kumulatif dihitung dengan:

```
P(i, j) = min( P(i, j-1), P(i-1, j-1), P(i-2, j-1) ) + C(i, j)
```

Tiga transisi tersebut adalah tetap di frame template yang sama, maju satu frame, dan melompati satu frame template. Tidak ada transisi vertikal, sehingga setiap frame input dikunjungi tepat satu kali dan biaya antar template bisa dibandingkan langsung.

Pembentukan averaged template per vowel:

1. Dari 5 rekaman template, rekaman dengan panjang median dipilih sebagai referensi.
2. Setiap rekaman lain di-align ke referensi dengan DTW, sehingga setiap frame referensi mendapat satu vektor pasangan.
3. Vektor-vektor yang ter-align dirata-ratakan per frame.

Pengenalan: `Audio -> MFCC -> DTW terhadap setiap template -> pilih cost terkecil -> vowel`.

### Sistem 2: Model Gaussian

Setiap vowel dimodelkan dengan satu Gaussian multivariat. Mean dan covariance penuh (39 x 39) dihitung dari semua frame kelima rekaman template vowel tersebut. Covariance ditambah `0.001 * I` agar tidak singular.

Cost input `X = {x_1, ..., x_T}` terhadap model vowel adalah rata-rata negative log-likelihood per frame:

```
d(x, m, C) = 1/2 log((2 pi)^D |C|) + 1/2 (x - m)^T C^-1 (x - m)
```

Model dengan cost terkecil dipilih sebagai hasil prediksi.

### Akurasi

```
Akurasi = jumlah prediksi benar / jumlah seluruh data uji x 100%
```
