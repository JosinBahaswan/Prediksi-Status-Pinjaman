"""Streamlit prototype – Prediksi Status Pinjaman.

Versi ini memuat model final yang SUDAH DILATIH (model.pkl) — berisi
KEDUA model (Random Forest & XGBoost) beserta metrik lengkap hasil
Nested CV — alih-alih menjalankan Nested CV + GridSearchCV setiap kali
aplikasi dibuka. Startup jadi hampir instan karena tidak ada training
sama sekali di runtime.

Staf kredit memasukkan data nasabah, lalu sistem menampilkan:
- Prediksi status pinjaman (Lunas / Gagal Bayar) dari salah satu atau
  kedua model sekaligus
- Probabilitas gagal bayar
- Metrik lengkap (Accuracy, Precision, Recall, F1, F2, AUC, FP/FN) &
  estimasi kerugian finansial
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────
MODEL_PATH = Path("model.pkl")
AVG_LOAN_AMOUNT = 33_042
RECOVERY_RATE = 0.30
PROFIT_MARGIN = 0.05

STATUS_PEKERJAAN_OPTIONS = ["Employed", "Self-Employed", "Student"]
TIPE_PRODUK_OPTIONS = ["Kartu Kredit", "Pinjaman Pribadi", "Kredit Berjalan"]
TUJUAN_PINJAMAN_OPTIONS = [
    "Bisnis", "Renovasi Rumah", "Konsolidasi Hutang",
    "Pendidikan", "Pribadi", "Medis",
]

# Rentang hyperparameter yang diuji di notebook (informasional saja —
# GridSearchCV sudah dijalankan sekali di notebook, tidak diulang di app)
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
}

# ── Load model (sekali, di-cache) ───────────────────────────────────

@st.cache_resource(show_spinner="Memuat model...")
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def prepare_single_input(row: dict, bundle: dict) -> pd.DataFrame:
    """Transform satu baris input mentah ke feature space yang sama
    dengan saat training, memakai imputer & encoder yang tersimpan
    di model.pkl (BUKAN di-fit ulang)."""
    df = pd.DataFrame([row])

    for col, val in bundle["num_impute_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in bundle["cat_impute_values"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    ohe = bundle["ohe_encoder"]
    ohe_arr = ohe.transform(df[bundle["cat_cols"]])
    ohe_cols = ohe.get_feature_names_out(bundle["cat_cols"]).tolist()

    df_final = pd.concat([
        df[bundle["num_cols"]].reset_index(drop=True),
        pd.DataFrame(ohe_arr, columns=ohe_cols),
    ], axis=1)

    df_final = df_final.reindex(columns=bundle["feature_names"], fill_value=0)
    return df_final.astype(float)


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Status Pinjaman",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #f1f5f9 !important; }
.main { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.stMarkdown *, p, span { color: #e0e0e0 !important; }
h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }

table { width: 100%; border-collapse: collapse; }
table th, table td {
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(0,0,0,0.2) !important;
    color: #e0e0e0 !important;
    padding: 10px;
}
thead tr th {
    background-color: rgba(255,255,255,0.1) !important;
    color: #f8fafc !important;
    font-weight: 600;
}
tbody tr:hover td { background-color: rgba(255,255,255,0.05) !important; }

.pred-card {
    border-radius: 16px; padding: 28px; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.35);
    backdrop-filter: blur(12px);
}
.pred-lunas {
    background: linear-gradient(135deg, rgba(16,185,129,.25), rgba(5,150,105,.15));
    border: 1px solid rgba(16,185,129,.4);
}
.pred-gagal {
    background: linear-gradient(135deg, rgba(239,68,68,.25), rgba(185,28,28,.15));
    border: 1px solid rgba(239,68,68,.4);
}
.metric-card {
    background: rgba(255,255,255,.06); border-radius: 12px;
    padding: 18px; text-align: center;
    border: 1px solid rgba(255,255,255,.08);
}
.metric-card h3 { color: #a78bfa !important; font-size: 14px; margin-bottom: 4px; }
.metric-card p { color: #f1f5f9 !important; font-size: 26px; font-weight: 700; margin: 0; }
.metric-card span { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model bundle di awal ───────────────────────────────────────
bundle = load_bundle()

if bundle is None:
    st.error(
        f"❌ File **{MODEL_PATH}** tidak ditemukan di direktori aplikasi. "
        "Jalankan cell 'Menyimpan Model Final' di notebook terlebih dahulu, "
        "lalu letakkan `model.pkl` di folder yang sama dengan `app.py` ini."
    )
    st.stop()

AVAILABLE_MODELS = list(bundle["models"].keys())  # ["Random Forest", "XGBoost"]
BEST_MODEL_NAME = bundle["best_model_name"]

# ── Sidebar – input form ─────────────────────────────────────────────
st.sidebar.markdown("## 📋 Data Nasabah")

usia = st.sidebar.number_input("Usia", 18, 70, 35)
status_pekerjaan = st.sidebar.selectbox("Status Pekerjaan", STATUS_PEKERJAAN_OPTIONS)
lama_bekerja = st.sidebar.number_input("Lama Bekerja (tahun)", 0.0, 40.0, 5.0, step=0.5)
pendapatan = st.sidebar.number_input("Pendapatan Tahunan ", 15_000, 250_000, 50_000, step=5000)
skor_kredit = st.sidebar.number_input("Skor Kredit", 300, 850, 650)
lama_riwayat = st.sidebar.number_input("Lama Riwayat Kredit (tahun)", 0.0, 30.0, 5.0, step=0.5)
aset_tabungan = st.sidebar.number_input("Aset / Tabungan ", 0, 300_000, 3_000, step=500)
hutang = st.sidebar.number_input("Hutang Saat Ini ", 0, 200_000, 10_000, step=1000)

st.sidebar.markdown("---")
tunggakan = st.sidebar.number_input("Tunggakan 2 Tahun Terakhir", 0, 10, 0)
catatan_negatif = st.sidebar.number_input("Catatan Negatif", 0, 5, 0)
tipe_produk = st.sidebar.selectbox("Tipe Produk", TIPE_PRODUK_OPTIONS)
tujuan = st.sidebar.selectbox("Tujuan Pinjaman", TUJUAN_PINJAMAN_OPTIONS)
jumlah_pinjaman = st.sidebar.number_input("Jumlah Pinjaman ", 500, 100_000, 20_000, step=1000)
suku_bunga = st.sidebar.number_input("Suku Bunga (%)", 6.0, 23.0, 15.0, step=0.5)

# Derived ratios
rasio_hutang = hutang / max(pendapatan, 1)
rasio_pinjaman = jumlah_pinjaman / max(pendapatan, 1)
rasio_pembayaran = (jumlah_pinjaman * suku_bunga / 100) / max(pendapatan, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Rasio Hutang / Pendapatan:** `{rasio_hutang:.3f}`")
st.sidebar.markdown(f"**Rasio Pinjaman / Pendapatan:** `{rasio_pinjaman:.3f}`")
st.sidebar.markdown(f"**Rasio Pembayaran / Pendapatan:** `{rasio_pembayaran:.3f}`")

model_choice = st.sidebar.radio(
    "Model",
    AVAILABLE_MODELS + ["Bandingkan Keduanya"],
    index=len(AVAILABLE_MODELS),  # default: Bandingkan Keduanya
)
predict_btn = st.sidebar.button("🔍 Prediksi Sekarang", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────
st.markdown("# 🏦 Sistem Prediksi Status Pinjaman")
st.markdown(f"##### Prototype untuk staf kredit — model dimuat langsung dari `model.pkl` (🏆 terbaik: **{BEST_MODEL_NAME}**)")

# Prepare input
input_row = {
    "usia": usia,
    "status_pekerjaan": status_pekerjaan,
    "lama_bekerja_tahun": lama_bekerja,
    "pendapatan_tahunan": pendapatan,
    "skor_kredit": skor_kredit,
    "lama_riwayat_kredit_tahun": lama_riwayat,
    "aset_tabungan": aset_tabungan,
    "hutang_saat_ini": hutang,
    "tunggakan_2thn_terakhir": tunggakan,
    "catatan_negatif": catatan_negatif,
    "tipe_produk": tipe_produk,
    "tujuan_pinjaman": tujuan,
    "jumlah_pinjaman": jumlah_pinjaman,
    "suku_bunga": suku_bunga,
    "rasio_hutang_terhadap_pendapatan": rasio_hutang,
    "rasio_pinjaman_terhadap_pendapatan": rasio_pinjaman,
    "rasio_pembayaran_terhadap_pendapatan": rasio_pembayaran,
}

if predict_btn:
    X_input = prepare_single_input(input_row, bundle)

    if model_choice == "Bandingkan Keduanya":
        models_to_run = [(name, bundle["models"][name]) for name in AVAILABLE_MODELS]
    else:
        models_to_run = [(model_choice, bundle["models"][model_choice])]

    cols = st.columns(len(models_to_run))
    for col, (mname, mdl) in zip(cols, models_to_run):
        pred = mdl.predict(X_input)[0]
        prob = mdl.predict_proba(X_input)[0]
        prob_gagal = prob[0]
        prob_lunas = prob[1]

        label = "✅ LUNAS" if pred == 1 else "❌ GAGAL BAYAR"
        css = "pred-lunas" if pred == 1 else "pred-gagal"

        with col:
            st.markdown(f"### {mname}")
            st.markdown(f"""
            <div class="pred-card {css}">
                <h2 style="margin:0;font-size:32px;color:{'#10b981' if pred==1 else '#ef4444'}">{label}</h2>
                <p style="margin-top:8px;font-size:15px;color:#94a3b8">
                    Probabilitas Gagal Bayar: <b>{prob_gagal:.1%}</b> &nbsp;|&nbsp;
                    Probabilitas Lunas: <b>{prob_lunas:.1%}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            loss = jumlah_pinjaman * (1 - RECOVERY_RATE) * prob_gagal
            st.markdown(f"""
            <div class="metric-card" style="margin-top:16px">
                <h3>💰 Estimasi Potensi Kerugian</h3>
                <p> {loss:,.0f}</p>
                <!-- <span style="font-size:12px;color:#94a3b8">= pinjaman × (1 − recovery) × P(gagal bayar)</span> -->
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model performance metrics (dari hasil Nested CV di notebook) ─
    st.markdown("## 📊 Performa Model")
    for mname in AVAILABLE_MODELS:
        m = bundle["metrics"][mname]
        star = " 🏆" if mname == BEST_MODEL_NAME else ""
        st.markdown(f"#### {mname}{star}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col_w, (label, key) in zip(
            [c1, c2, c3, c4, c5, c6],
            [("Accuracy", "accuracy"), ("Precision", "precision"),
             ("Recall", "recall"), ("F1-Score", "f1_score"),
             ("F2-Score", "f2_score"), ("AUC", "auc")],
        ):
            col_w.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p>{m['mean'][key]:.4f}</p>
                <span style="font-size:11px;color:#64748b">±{m['std'][key]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        fp_loss = m["total_fp"] * AVG_LOAN_AMOUNT * (1 - RECOVERY_RATE)
        fn_cost = m["total_fn"] * AVG_LOAN_AMOUNT * PROFIT_MARGIN
        st.markdown(f"""
        <!--
        <div class="metric-card" style="margin:12px 0 24px 0">
            <h3>💸 Estimasi Kerugian Finansial </h3>
            <p> {fp_loss + fn_cost:,.0f}</p>
            <span style="font-size:12px;color:#94a3b8">
                FP ({m['total_fp']:,}×) = {fp_loss:,.0f} &nbsp;|&nbsp;
                FN ({m['total_fn']:,}×) = {fn_cost:,.0f}
            </span>
        </div>
        -->
        """, unsafe_allow_html=True)

    # ── Hyperparameter model final ──────────────────────────────────
    st.markdown("## 🔧 Hyperparameter Model Final")
    col_rf, col_xgb = st.columns(2)
    for col_w, mname, grid in [(col_rf, "Random Forest", RF_PARAM_GRID), (col_xgb, "XGBoost", XGB_PARAM_GRID)]:
        with col_w:
            st.markdown(f"#### {mname}")
            best_p = bundle["metrics"][mname]["params_repr"]
            rows = [{"Hyperparameter": k, "Nilai Terpilih": v, "Rentang Diuji": str(grid.get(k, "—"))}
                    for k, v in best_p.items() if k in grid]
            st.table(pd.DataFrame(rows))
            total = 1
            for v in grid.values():
                total *= len(v)
            st.caption(f"Total kombinasi diuji: **{total}**")

else:
    # Landing state
    st.info("👈 Isi data nasabah di sidebar lalu klik **Prediksi Sekarang**")

    st.markdown("## ℹ️ Tentang Sistem")
    c1, c2, c3 = st.columns(3)
    for col_w, icon, title, desc in [
        (c1, "🌲", "Random Forest", "Ensemble bagging dari banyak decision tree"),
        (c2, "🚀", "XGBoost", "Gradient boosting yang dioptimasi untuk kecepatan"),
        (c3, "⚡", "Instan", "Kedua model sudah dilatih sebelumnya (model.pkl) — tidak ada training saat aplikasi dibuka"),
    ]:
        col_w.markdown(f"""
        <div class="metric-card">
            <p style="font-size:36px;margin-bottom:4px">{icon}</p>
            <h3 style="font-size:16px !important">{title}</h3>
            <p style="font-size:13px;font-weight:400;color:#94a3b8">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 Fitur yang Digunakan")
    st.markdown("""
    | No | Fitur | Tipe |
    |---|---|---|
    | 1 | Usia | Numerik |
    | 2 | Status Pekerjaan | Kategorikal (One-Hot) |
    | 3 | Lama Bekerja (tahun) | Numerik |
    | 4 | Pendapatan Tahunan | Numerik |
    | 5 | Skor Kredit | Numerik |
    | 6 | Lama Riwayat Kredit | Numerik |
    | 7 | Aset / Tabungan | Numerik |
    | 8 | Hutang Saat Ini | Numerik |
    | 9 | Tunggakan 2 Tahun Terakhir | Numerik |
    | 10 | Catatan Negatif | Numerik |
    | 11 | Tipe Produk | Kategorikal (One-Hot) |
    | 12 | Tujuan Pinjaman | Kategorikal (One-Hot) |
    | 13 | Jumlah Pinjaman | Numerik |
    | 14 | Suku Bunga | Numerik |
    | 15 | Rasio Hutang / Pendapatan | Numerik |
    | 16 | Rasio Pinjaman / Pendapatan | Numerik |
    | 17 | Rasio Pembayaran / Pendapatan | Numerik |
    """)