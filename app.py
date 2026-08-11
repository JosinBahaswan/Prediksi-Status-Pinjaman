"""Streamlit prototype – Prediksi Status Pinjaman.

Versi ini memuat model final yang SUDAH DILATIH (model.pkl) — berisi
KEDUA model (Random Forest & XGBoost) beserta metrik lengkap hasil
Nested CV — alih-alih menjalankan Nested CV + GridSearchCV setiap kali
aplikasi dibuka. Startup jadi hampir instan karena tidak ada training
sama sekali di runtime.

Model yang disimpan berasal dari fold dengan F2-Score tertinggi
dalam proses Nested CV (bukan fit ulang pada seluruh data).

Staf kredit memasukkan data nasabah, lalu sistem menampilkan:
- Prediksi status pinjaman (Lunas / Gagal Bayar) dari salah satu atau
  kedua model sekaligus
- Probabilitas gagal bayar
- SHAP waterfall plot lokal (penjelasan kenapa nasabah ditolak/disetujui)
- Metrik lengkap (Accuracy, Precision, Recall, F1, F2, AUC)
"""

import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────
MODEL_PATH = Path("model.pkl")

STATUS_REKENING_OPTIONS = ["0-200_DM", "diatas_200_DM", "dibawah_0_DM", "tidak_ada"]
RIWAYAT_KREDIT_OPTIONS = ["kritis/ada_kredit_lain", "lancar/tidak_ada_kredit", "lancar_hingga_kini", "pernah_telat", "semua_lancar_di_bank_ini"]
TUJUAN_PINJAMAN_OPTIONS = ["alat_rumah_tangga", "bisnis", "lainnya", "liburan", "mobil_baru", "mobil_bekas", "pelatihan_ulang", "perabot/peralatan", "perbaikan", "radio/tv"]
ASET_TABUNGAN_OPTIONS = ["100-500_DM", "500-1000_DM", "diatas_1000_DM", "dibawah_100_DM", "tidak_diketahui/tidak_ada"]
LAMA_BEKERJA_OPTIONS = ["1-4_thn", "4-7_thn", "diatas_7_thn", "dibawah_1_thn", "menganggur"]
RASIO_CICILAN_OPTIONS = ["20_sd_25_persen", "25_sd_35_persen", "diatas_35_persen", "dibawah_20_persen"]
STATUS_PERSONAL_KELAMIN_OPTIONS = ["pria_cerai/pisah", "pria_menikah/duda", "wanita_lajang", "wanita_menikah_atau_pria_lajang"]
LAMA_TINGGAL_OPTIONS = ["1-4_thn", "4-7_thn", "diatas_7_thn", "dibawah_1_thn"]
KEPEMILIKAN_HARTA_OPTIONS = ["asuransi_jiwa/tabungan", "mobil/lainnya", "real_estate", "tidak_ada/tidak_diketahui"]
CICILAN_LAIN_OPTIONS = ["bank", "tidak_ada", "toko"]
PERUMAHAN_OPTIONS = ["gratis", "milik_sendiri", "sewa"]
PEKERJAAN_OPTIONS = ["karyawan_ahli/pejabat", "manajer/wiraswasta", "menganggur/tidak_ahli_non-residen", "tidak_ahli_residen"]

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


def prepare_single_input(row: dict, bundle: dict, model_name: str) -> pd.DataFrame:
    """Transform satu baris input mentah ke feature space yang sama
    dengan saat training, memakai imputer & encoder yang tersimpan
    di model.pkl (BUKAN di-fit ulang).

    OHE encoder, impute values, dan feature_names bisa disimpan per-model
    (dict keyed by model_name) atau sebagai nilai tunggal — keduanya didukung.
    """
    df = pd.DataFrame([row])

    # Ambil impute values — bisa dict per-model atau langsung dict fitur
    num_impute = bundle["num_impute_values"]
    cat_impute = bundle["cat_impute_values"]
    if isinstance(num_impute, dict) and model_name in num_impute:
        num_impute = num_impute[model_name]
    if isinstance(cat_impute, dict) and model_name in cat_impute:
        cat_impute = cat_impute[model_name]

    for col, val in num_impute.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in cat_impute.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Ambil OHE encoder — bisa dict per-model atau encoder tunggal
    ohe = bundle["ohe_encoder"]
    if isinstance(ohe, dict):
        ohe = ohe[model_name]

    ohe_arr = ohe.transform(df[bundle["cat_cols"]])
    ohe_cols = ohe.get_feature_names_out(bundle["cat_cols"]).tolist()

    df_final = pd.concat([
        df[bundle["num_cols"]].reset_index(drop=True),
        pd.DataFrame(ohe_arr, columns=ohe_cols),
    ], axis=1)

    # Ambil feature_names — bisa dict per-model atau list tunggal
    feat_names = bundle["feature_names"]
    if isinstance(feat_names, dict):
        feat_names = feat_names[model_name]

    df_final = df_final.reindex(columns=feat_names, fill_value=0)
    return df_final.astype(float)


@st.cache_resource(show_spinner="Menyiapkan SHAP explainer...")
def get_shap_explainer(_model, model_name: str):
    """Buat TreeExplainer untuk model yang diberikan (di-cache per model_name)."""
    return shap.TreeExplainer(_model)


def plot_shap_waterfall(explainer, X_input: pd.DataFrame, model_name: str, pred: int):
    """Buat SHAP waterfall plot untuk satu nasabah."""
    shap_obj = explainer(X_input)

    # RF mengembalikan Explanation 3D (samples, features, classes) → ambil kelas 1 (Lunas)
    if len(shap_obj.shape) == 3:
        shap_single = shap_obj[0, :, 1]
    else:
        shap_single = shap_obj[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.waterfall_plot(shap_single, max_display=10, show=False)

    label = "LUNAS" if pred == 1 else "GAGAL BAYAR"
    ax.set_title(
        f"SHAP — Alasan Prediksi '{label}' ({model_name})",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff22")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    ax.title.set_color("#f8fafc")
    plt.tight_layout()
    return fig


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

.shap-section {
    background: rgba(255,255,255,.04); border-radius: 12px;
    padding: 12px 16px; margin-top: 16px;
    border: 1px solid rgba(167,139,250,.25);
}
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

usia = st.sidebar.number_input("Usia", 18, 100, 35)
durasi_pinjaman_bulan = st.sidebar.number_input("Durasi Pinjaman (bulan)", 1, 120, 24)
jumlah_pinjaman = st.sidebar.number_input("Jumlah Pinjaman (DM)", 100, 100000, 2000, step=500)

st.sidebar.markdown("---")
status_rekening = st.sidebar.selectbox("Status Rekening", STATUS_REKENING_OPTIONS)
riwayat_kredit = st.sidebar.selectbox("Riwayat Kredit", RIWAYAT_KREDIT_OPTIONS)
tujuan_pinjaman = st.sidebar.selectbox("Tujuan Pinjaman", TUJUAN_PINJAMAN_OPTIONS)
aset_tabungan = st.sidebar.selectbox("Aset Tabungan", ASET_TABUNGAN_OPTIONS)
lama_bekerja = st.sidebar.selectbox("Lama Bekerja", LAMA_BEKERJA_OPTIONS)
rasio_cicilan = st.sidebar.selectbox("Rasio Cicilan", RASIO_CICILAN_OPTIONS)
status_personal_dan_kelamin = st.sidebar.selectbox("Status Personal & Kelamin", STATUS_PERSONAL_KELAMIN_OPTIONS)
lama_tinggal = st.sidebar.selectbox("Lama Tinggal", LAMA_TINGGAL_OPTIONS)
kepemilikan_harta = st.sidebar.selectbox("Kepemilikan Harta", KEPEMILIKAN_HARTA_OPTIONS)
cicilan_lain = st.sidebar.selectbox("Cicilan Lain", CICILAN_LAIN_OPTIONS)
perumahan = st.sidebar.selectbox("Perumahan", PERUMAHAN_OPTIONS)
pekerjaan = st.sidebar.selectbox("Pekerjaan", PEKERJAAN_OPTIONS)

st.sidebar.markdown("---")
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
    "durasi_pinjaman_bulan": durasi_pinjaman_bulan,
    "jumlah_pinjaman": jumlah_pinjaman,
    "usia": usia,
    "status_rekening": status_rekening,
    "riwayat_kredit": riwayat_kredit,
    "tujuan_pinjaman": tujuan_pinjaman,
    "aset_tabungan": aset_tabungan,
    "lama_bekerja": lama_bekerja,
    "rasio_cicilan": rasio_cicilan,
    "status_personal_dan_kelamin": status_personal_dan_kelamin,
    "lama_tinggal": lama_tinggal,
    "kepemilikan_harta": kepemilikan_harta,
    "cicilan_lain": cicilan_lain,
    "perumahan": perumahan,
    "pekerjaan": pekerjaan
}

if predict_btn:
    if model_choice == "Bandingkan Keduanya":
        models_to_run = [(name, bundle["models"][name]) for name in AVAILABLE_MODELS]
    else:
        models_to_run = [(model_choice, bundle["models"][model_choice])]

    cols = st.columns(len(models_to_run))
    for col, (mname, mdl) in zip(cols, models_to_run):
        X_input = prepare_single_input(input_row, bundle, mname)
        pred = mdl.predict(X_input)[0]
        prob = mdl.predict_proba(X_input)[0]
        prob_gagal = prob[0]
        prob_lunas = prob[1]

        label = "\u2705 LUNAS" if pred == 1 else "\u274c GAGAL BAYAR"
        css = "pred-lunas" if pred == 1 else "pred-gagal"

        with col:
            st.markdown(f"### {mname}")
            color_hex = '#10b981' if pred == 1 else '#ef4444'
            st.markdown(
                f'<div class="pred-card {css}">'
                f'<h2 style="margin:0;font-size:32px;color:{color_hex}">{label}</h2>'
                f'<p style="margin-top:8px;font-size:15px;color:#94a3b8">'
                f'Probabilitas Gagal Bayar: <b>{prob_gagal:.1%}</b> &nbsp;|&nbsp;'
                f'Probabilitas Lunas: <b>{prob_lunas:.1%}</b></p></div>',
                unsafe_allow_html=True,
            )

            # ── SHAP waterfall plot (penjelasan lokal per nasabah) ──
            alasan = "disetujui" if pred == 1 else "ditolak"
            st.markdown(
                f'<div class="shap-section"><b>\U0001f50d Mengapa nasabah ini {alasan}?</b></div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Menghitung SHAP..."):
                explainer = get_shap_explainer(mdl, mname)
                fig = plot_shap_waterfall(explainer, X_input, mname, pred)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.caption(
                "\U0001f4cc Batang **merah** (kanan) = fitur yang **meningkatkan** probabilitas ke prediksi ini. "
                "Batang **biru** (kiri) = fitur yang **menurunkan** probabilitas."
            )

    st.markdown("---")

    # ── Model performance metrics (dari hasil Nested CV di notebook) ─
    st.markdown("## \U0001f4ca Performa Model")
    for mname in AVAILABLE_MODELS:
        m = bundle["metrics"][mname]
        star = " \U0001f3c6" if mname == BEST_MODEL_NAME else ""
        st.markdown(f"#### {mname}{star}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col_w, (label, key) in zip(
            [c1, c2, c3, c4, c5, c6],
            [("Accuracy", "accuracy"), ("Precision", "precision"),
             ("Recall", "recall"), ("F1-Score", "f1_score"),
             ("F2-Score", "f2_score"), ("AUC", "auc")],
        ):
            col_w.markdown(
                f'<div class="metric-card">'
                f'<h3>{label}</h3>'
                f'<p>{m["mean"][key]:.4f}</p>'
                f'<span style="font-size:11px;color:#64748b">\u00b1{m["std"][key]:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

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
    st.info("\U0001f448 Isi data nasabah di sidebar lalu klik **Prediksi Sekarang**")

    st.markdown("## \u2139\ufe0f Tentang Sistem")
    c1, c2, c3 = st.columns(3)
    for col_w, icon, title, desc in [
        (c1, "\U0001f332", "Random Forest", "Ensemble bagging dari banyak decision tree — model dari fold F2 terbaik"),
        (c2, "\U0001f680", "XGBoost", "Gradient boosting yang dioptimasi untuk kecepatan — model dari fold F2 terbaik"),
        (c3, "\U0001f52c", "SHAP Explanation", "Penjelasan transparan: fitur mana yang paling berpengaruh pada keputusan"),
    ]:
        col_w.markdown(
            f'<div class="metric-card">'
            f'<p style="font-size:36px;margin-bottom:4px">{icon}</p>'
            f'<h3 style="font-size:16px !important">{title}</h3>'
            f'<p style="font-size:13px;font-weight:400;color:#94a3b8">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📝 Fitur yang Digunakan")
    st.markdown("""
    | No | Fitur | Tipe |
    |---|---|---|
    | 1 | Usia | Numerik |
    | 2 | Durasi Pinjaman (bulan) | Numerik |
    | 3 | Jumlah Pinjaman | Numerik |
    | 4 | Status Rekening | Kategorikal (One-Hot) |
    | 5 | Riwayat Kredit | Kategorikal (One-Hot) |
    | 6 | Tujuan Pinjaman | Kategorikal (One-Hot) |
    | 7 | Aset Tabungan | Kategorikal (One-Hot) |
    | 8 | Lama Bekerja | Kategorikal (One-Hot) |
    | 9 | Rasio Cicilan | Kategorikal (One-Hot) |
    | 10 | Status Personal & Kelamin | Kategorikal (One-Hot) |
    | 11 | Lama Tinggal | Kategorikal (One-Hot) |
    | 12 | Kepemilikan Harta | Kategorikal (One-Hot) |
    | 13 | Cicilan Lain | Kategorikal (One-Hot) |
    | 14 | Perumahan | Kategorikal (One-Hot) |
    | 15 | Pekerjaan | Kategorikal (One-Hot) |
    """)