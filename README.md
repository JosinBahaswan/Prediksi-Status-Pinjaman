# 🏦 Prediksi Status Pinjaman (German Credit Data)

Sistem Prediksi Status Pinjaman adalah prototipe aplikasi berbasis web yang dibangun menggunakan **Streamlit** dan **Machine Learning**. Aplikasi ini dirancang untuk membantu staf kredit dalam mengidentifikasi risiko kelayakan nasabah (Lunas vs. Gagal Bayar) secara presisi, cepat, dan terukur.

---

## 🎯 Fitur Utama

- **Prediksi Real-time**: Mendukung evaluasi risiko pinjaman menggunakan model **Random Forest** dan **XGBoost**.
- **Probabilitas Risiko**: Menampilkan tingkat kepercayaan model serta estimasi persentase risiko gagal bayar.
- **Explainable AI (SHAP)**: Dilengkapi dengan visualisasi *SHAP Waterfall Plot* lokal untuk menjelaskan fitur-fitur utama yang memengaruhi keputusan penolakan atau persetujuan kredit.
- **Evaluasi Metrik**: Menyediakan performa metrik lengkap (*Accuracy*, *Precision*, *Recall*, *F1-Score*, *F2-Score*, dan *AUC*) hasil uji *Nested Cross-Validation*.

---

## 📁 Struktur Direktori

```text
.
├── data/                                             # Dataset kredit (German Credit Data)
├── diagram/                                          # Diagram arsitektur dan alur sistem
├── depracated/                                       # Berkas / eksperimen versi sebelumnya
├── app.py                                            # Aplikasi utama Streamlit
├── model.pkl                                         # Bundle model ML final (Random Forest & XGBoost)
├── requirements.txt                                  # Daftar dependensi Python
├── skripsi_prediksi_german_credit_runv8_generate.ipynb # Notebook pelatihan model & EDA
├── .gitignore                                        # Konfigurasi berkas yang diabaikan oleh Git
└── README.md                                         # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Prasyarat
- **Python 3.10** atau versi lebih baru.
- Git (opsional).

### 2. Memasang Virtual Environment (Disarankan)

**Di Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Di Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Di Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Menginstal Dependensi

Setelah virtual environment aktif, jalankan:
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Aplikasi Web

Jalankan perintah berikut untuk mengoperasikan aplikasi Streamlit:
```bash
streamlit run app.py
```

Aplikasi secara otomatis akan terbuka di peramban web (*browser*) pada alamat `http://localhost:8501`.

---

## 📊 Model & Metrik

Model ML pada proyek ini dilatih dengan mengoptimalkan **F2-Score** untuk meminimalkan risiko *False Negative* (nasabah berisiko gagal bayar yang salah terprediksi sebagai lunas). 

- **Algoritma**: Random Forest Classifier & XGBoost Classifier
- **Teknik Pemodelan**: Handling Imbalanced Data, One-Hot Encoding, Median/Mode Imputation, Nested Cross-Validation & Hyperparameter Tuning.
- **Interpretabilitas**: SHAP (SHapley Additive exPlanations).

---

## 📝 Catatan Tambahan
Dibuat untuk keperluan Skripsi / Tugas Akhir Prediksi Status Pinjaman Kredit.
