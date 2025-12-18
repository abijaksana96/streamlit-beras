# 🌾 Dashboard Prediksi Harga Beras Premium Papua

Dashboard interaktif untuk menganalisis dan memprediksi harga beras premium di Papua menggunakan model LSTM (Long Short-Term Memory).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Fitur Utama](#fitur-utama)
- [Struktur Proyek](#struktur-proyek)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Dataset](#dataset)
- [Model LSTM](#model-lstm)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Lisensi](#lisensi)

## 📖 Tentang Proyek

Dashboard ini dirancang untuk menganalisis dan memprediksi harga beras premium di Papua menggunakan teknologi deep learning. Dengan memanfaatkan model LSTM (Long Short-Term Memory), dashboard ini dapat memberikan prediksi harga beras berdasarkan data historis.

### Latar Belakang

Beras merupakan komoditas pangan utama di Indonesia. Fluktuasi harga beras, khususnya di wilayah Papua, dapat mempengaruhi perekonomian masyarakat. Dashboard ini hadir sebagai solusi untuk:

- Memantau trend harga beras premium
- Menganalisis pola pergerakan harga
- Memprediksi harga beras di masa depan
- Mendukung pengambilan keputusan berbasis data

## ✨ Fitur Utama

### 🏠 Beranda
- Ringkasan statistik harga beras (minimum, maksimum, rata-rata)
- Preview data terbaru
- Informasi periode data

### 📈 Analisis Data
- Visualisasi trend harga beras premium
- Grafik distribusi harga (histogram)
- Box plot untuk analisis sebaran data
- Statistik deskriptif lengkap

### 🤖 Model Prediksi
- Prediksi harga menggunakan model LSTM
- Visualisasi perbandingan prediksi vs data aktual
- Informasi arsitektur model
- Parameter training

### 📋 Evaluasi Model
- Metrik evaluasi (RMSE, MAE, MAPE)
- Interpretasi tingkat akurasi
- Analisis residual
- Plot residual dan distribusinya

## 📁 Struktur Proyek

```
streamlit-beras/
├── dashboard.py              # Aplikasi Streamlit utama
├── Beras Premium Train.csv   # Dataset harga beras
├── lstm_model.h5             # Model LSTM yang sudah dilatih
├── requirements.txt          # Daftar dependensi Python
└── README.md                 # Dokumentasi proyek
```

## 💻 Prasyarat

Sebelum menjalankan proyek ini, pastikan Anda memiliki:

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Git (opsional, untuk clone repository)

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/abijaksana96/streamlit-beras.git
cd streamlit-beras
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

## ▶️ Cara Menjalankan

Jalankan aplikasi Streamlit dengan perintah:

```bash
streamlit run dashboard.py
```

Aplikasi akan terbuka secara otomatis di browser pada alamat `http://localhost:8501`.

## 📚 Panduan Penggunaan

### Navigasi

Gunakan sidebar di sebelah kiri untuk berpindah antar halaman:

1. **🏠 Beranda** - Halaman utama dengan ringkasan statistik
2. **📈 Analisis Data** - Eksplorasi dan visualisasi data
3. **🤖 Model Prediksi** - Menjalankan prediksi dengan model LSTM
4. **📋 Evaluasi Model** - Melihat performa model

### Menjalankan Prediksi

1. Pilih menu **🤖 Model Prediksi** di sidebar
2. Klik tombol **🚀 Jalankan Prediksi**
3. Tunggu proses prediksi selesai
4. Lihat hasil prediksi dalam bentuk grafik

### Melihat Evaluasi Model

1. Pastikan sudah menjalankan prediksi terlebih dahulu
2. Pilih menu **📋 Evaluasi Model** di sidebar
3. Lihat metrik evaluasi dan analisis residual

## 📊 Dataset

### Sumber Data

Dataset berisi harga beras premium harian dari berbagai provinsi di Indonesia.

### Informasi Dataset

| Atribut | Deskripsi |
|---------|-----------|
| Periode | 1 Januari 2022 - September 2024 |
| Provinsi | 34 provinsi di Indonesia |
| Fokus Analisis | Provinsi Papua |
| Format | CSV (Comma Separated Values) |

### Kolom Dataset

- `Date`: Tanggal pencatatan harga
- `Papua`: Harga beras premium di Papua (Rp/kg)
- Kolom lainnya: Harga di provinsi-provinsi lain

## 🧠 Model LSTM

### Arsitektur Model

Model LSTM yang digunakan memiliki arsitektur sebagai berikut:

| Layer | Unit | Keterangan |
|-------|------|------------|
| LSTM Layer 1 | 256 | Return sequences |
| Dropout 1 | 0.1 | Regularization |
| LSTM Layer 2 | 128 | Return sequences |
| Dropout 2 | 0.1 | Regularization |
| LSTM Layer 3 | 64 | - |
| Dropout 3 | 0.1 | Regularization |
| Dense | 32 | Hidden layer |
| Dense | 1 | Output layer |

### Parameter Training

| Parameter | Nilai |
|-----------|-------|
| Timesteps | 60 hari |
| Epochs | 200 (dengan early stopping) |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |

### Preprocessing Data

1. **Normalisasi**: MinMaxScaler (range 0-1)
2. **Sequence Creation**: 60 timesteps
3. **Train-Test Split**: 30 data terakhir untuk testing

## 🛠️ Teknologi yang Digunakan

### Framework & Library Utama

| Teknologi | Versi | Fungsi |
|-----------|-------|--------|
| Python | 3.8+ | Bahasa pemrograman |
| Streamlit | 1.49.1 | Web framework |
| TensorFlow | 2.20.0 | Deep learning |
| Keras | 3.11.3 | Neural network API |
| Pandas | 2.3.2 | Manipulasi data |
| NumPy | 2.3.2 | Komputasi numerik |
| Plotly | 6.3.0 | Visualisasi interaktif |
| Matplotlib | 3.10.5 | Visualisasi statis |
| Seaborn | 0.13.2 | Visualisasi statistik |
| Scikit-learn | 1.7.1 | Machine learning tools |

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
