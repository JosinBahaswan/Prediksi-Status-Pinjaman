"""Streamlit prototype – Prediksi Status Pinjaman.

Staf kredit memasukkan data nasabah, lalu sistem menampilkan:
- Prediksi status pinjaman (Lunas / Gagal Bayar)
- Probabilitas gagal bayar
- F2-Score model & estimasi kerugian finansial
"""

import warnings, pickle, os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────
DATA_PATH = Path("data/loan_data.csv")
MODEL_CACHE = Path("trained_model.pkl")
TARGET_COL = "status_pinjaman"
DROP_COLS = ["id_pelanggan", "gagal_bayar_tercatat"]
RANDOM_STATE = 42
INNER_SPLITS = 3
AVG_LOAN_AMOUNT = 33_042
RECOVERY_RATE = 0.30
PROFIT_MARGIN = 0.05

STATUS_PEKERJAAN_OPTIONS = ["Employed", "Self-Employed", "Student"]
TIPE_PRODUK_OPTIONS = ["Kartu Kredit", "Pinjaman Pribadi", "Kredit Berjalan"]
TUJUAN_PINJAMAN_OPTIONS = [
    "Bisnis", "Renovasi Rumah", "Konsolidasi Hutang",
    "Pendidikan", "Pribadi", "Medis",
]

RF_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}
XGB_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8],
}

# ── Helpers ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def preprocess_ohe(X: pd.DataFrame):
    """Full-dataset OHE for training (impute → one-hot)."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    Xp = X.copy()
    num_imp = SimpleImputer(strategy="median")
    Xp[num_cols] = num_imp.fit_transform(Xp[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    Xp[cat_cols] = cat_imp.fit_transform(Xp[cat_cols])
    Xp = pd.get_dummies(Xp, columns=cat_cols, drop_first=False)
    return Xp, num_imp, cat_imp, num_cols, cat_cols


@st.cache_resource(show_spinner="Melatih model (sekali saja)…")
def train_models():
    X_raw, y = load_data()
    metrics = compute_cv_metrics(X_raw, y)
    rf_params = metrics["rf"]["params_repr"]
    xgb_params = metrics["xgb"]["params_repr"]

    X_proc, num_imp, cat_imp, num_cols, cat_cols = preprocess_ohe(X_raw)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_proc, y)

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_sm, y_sm)

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_sm, y_sm)

    return {
        "rf": rf, "xgb": xgb,
        "num_imp": num_imp, "cat_imp": cat_imp,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "feature_cols": X_proc.columns.tolist(),
        "metrics": metrics,
    }


def _representative_params(best_params_per_fold: list[dict], extra_params: dict) -> dict:
    if not best_params_per_fold:
        return extra_params.copy()
    params_repr = {}
    for key in best_params_per_fold[0]:
        vals = [p[key] for p in best_params_per_fold]
        params_repr[key] = Counter(vals).most_common(1)[0][0]
    params_repr.update(extra_params)
    return params_repr


def _nested_cv_for_model(
    X_raw: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    param_grid: dict,
    n_splits: int = 10,
    inner_splits: int = INNER_SPLITS,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    metrics_rows = []
    total_fp = 0
    total_fn = 0
    best_params_per_fold = []

    if model_name == "rf":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        extra_params = {"random_state": RANDOM_STATE, "n_jobs": -1}
    else:
        base_model = XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        extra_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

    for train_idx, val_idx in skf.split(X_raw, y):
        X_train_raw = X_raw.iloc[train_idx].copy()
        X_val_raw = X_raw.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        # Imputation per fold
        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            X_train_raw[num_cols] = num_imputer.fit_transform(X_train_raw[num_cols])
            X_val_raw[num_cols] = num_imputer.transform(X_val_raw[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train_raw[cat_cols] = cat_imputer.fit_transform(X_train_raw[cat_cols])
            X_val_raw[cat_cols] = cat_imputer.transform(X_val_raw[cat_cols])

        # OHE per fold (fit only on train)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_ohe = ohe.fit_transform(X_train_raw[cat_cols])
        X_val_ohe = ohe.transform(X_val_raw[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols)

        X_train_fold = pd.concat(
            [
                X_train_raw[num_cols].reset_index(drop=True),
                pd.DataFrame(X_train_ohe, columns=ohe_names),
            ],
            axis=1,
        ).astype(float)
        X_val_fold = pd.concat(
            [
                X_val_raw[num_cols].reset_index(drop=True),
                pd.DataFrame(X_val_ohe, columns=ohe_names),
            ],
            axis=1,
        ).astype(float)

        pipeline_inner = Pipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", base_model),
            ]
        )
        grid = GridSearchCV(
            pipeline_inner,
            param_grid,
            scoring="f1",
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(X_train_fold, y_train_fold)

        best_p = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        best_params_per_fold.append(best_p)

        smote_fold = SMOTE(random_state=RANDOM_STATE)
        X_train_smote, y_train_smote = smote_fold.fit_resample(X_train_fold, y_train_fold)

        if model_name == "rf":
            model = RandomForestClassifier(**best_p, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            model = XGBClassifier(
                **best_p,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        model.fit(X_train_smote, y_train_smote)

        y_pred = model.predict(X_val_fold)
        y_prob = model.predict_proba(X_val_fold)[:, 1]

        acc = accuracy_score(y_val_fold, y_pred)
        prec = precision_score(y_val_fold, y_pred)
        rec = recall_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)
        f2 = fbeta_score(y_val_fold, y_pred, beta=2, zero_division=0)
        auc = roc_auc_score(y_val_fold, y_prob)

        cm = confusion_matrix(y_val_fold, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_fp += fp
        total_fn += fn

        metrics_rows.append({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f2": f2,
            "auc": auc,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    summary = {
        "mean": metrics_df[["accuracy", "precision", "recall", "f1", "f2", "auc"]].mean(),
        "std": metrics_df[["accuracy", "precision", "recall", "f1", "f2", "auc"]].std(),
        "total_fn": int(total_fn),
        "total_fp": int(total_fp),
        "params_repr": _representative_params(best_params_per_fold, extra_params),
    }
    return summary


def compute_cv_metrics(X_raw, y, n_splits=10, inner_splits=INNER_SPLITS):
    return {
        "rf": _nested_cv_for_model(
            X_raw, y, model_name="rf", param_grid=RF_PARAM_GRID,
            n_splits=n_splits, inner_splits=inner_splits,
        ),
        "xgb": _nested_cv_for_model(
            X_raw, y, model_name="xgb", param_grid=XGB_PARAM_GRID,
            n_splits=n_splits, inner_splits=inner_splits,
        ),
    }


def prepare_single_input(row: dict, bundle: dict) -> pd.DataFrame:
    """Transform a single input row into the same feature space as training."""
    df = pd.DataFrame([row])
    num_cols = bundle["num_cols"]
    cat_cols = bundle["cat_cols"]
    df[num_cols] = bundle["num_imp"].transform(df[num_cols])
    df[cat_cols] = bundle["cat_imp"].transform(df[cat_cols])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    for c in bundle["feature_cols"]:
        if c not in df.columns:
            df[c] = 0.0
    df = df[bundle["feature_cols"]].astype(float)
    return df


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

/* Force background and global text color */
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #f1f5f9 !important; }
.main { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Force text colors for tables, markdown, and paragraphs */
.stMarkdown *, p, span { color: #e0e0e0 !important; }
h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }

/* Table styling for dark theme */
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

/* Custom Cards */
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

model_choice = st.sidebar.radio("Model", ["Random Forest", "XGBoost", "Bandingkan Keduanya"])
predict_btn = st.sidebar.button("🔍 Prediksi Sekarang", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────
st.markdown("# 🏦 Sistem Prediksi Status Pinjaman")
st.markdown("##### Prototype untuk staf kredit — masukkan data di sidebar lalu klik **Prediksi**")

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
    bundle = train_models()
    X_input = prepare_single_input(input_row, bundle)

    models_to_run = []
    if model_choice == "Random Forest":
        models_to_run = [("Random Forest", bundle["rf"])]
    elif model_choice == "XGBoost":
        models_to_run = [("XGBoost", bundle["xgb"])]
    else:
        models_to_run = [("Random Forest", bundle["rf"]), ("XGBoost", bundle["xgb"])]

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

            # Financial loss estimate for this individual
            loss = jumlah_pinjaman * (1 - RECOVERY_RATE) * prob_gagal
            st.markdown(f"""
            <div class="metric-card" style="margin-top:16px">
                <h3>💰 Estimasi Potensi Kerugian</h3>
                <p> {loss:,.0f}</p>
                <span style="font-size:12px;color:#94a3b8">= pinjaman × (1 − recovery) × P(gagal bayar)</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model performance metrics ─────────────────────────────────
    st.markdown("## 📊 Performa Model (10-Fold CV)")
    metrics = bundle["metrics"]
    for mkey, mname in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
        m = metrics[mkey]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        st.markdown(f"#### {mname}")
        for col_w, (label, key) in zip(
            [c1, c2, c3, c4, c5, c6],
            [("Accuracy", "accuracy"), ("Precision", "precision"),
             ("Recall", "recall"), ("F1-Score", "f1"),
             ("F2-Score", "f2"), ("AUC", "auc")],
        ):
            col_w.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p>{m['mean'][key]:.4f}</p>
                <span style="font-size:11px;color:#64748b">±{m['std'][key]:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        # Financial loss (FP = default loss, FN = missed profit)
        fp_loss = m["total_fp"] * AVG_LOAN_AMOUNT * (1 - RECOVERY_RATE)
        fn_cost = m["total_fn"] * AVG_LOAN_AMOUNT * PROFIT_MARGIN
        st.markdown(f"""
        <div class="metric-card" style="margin:12px 0 24px 0">
            <h3>💸 Estimasi Kerugian Finansial (seluruh fold)</h3>
            <p> {fp_loss + fn_cost:,.0f}</p>
            <span style="font-size:12px;color:#94a3b8">
                FP ({m['total_fp']:,}×) = {fp_loss:,.0f} &nbsp;|&nbsp;
                FN ({m['total_fn']:,}×) = {fn_cost:,.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Hyperparameter search table ───────────────────────────────
    st.markdown("## 🔧 Rentang Hyperparameter yang Diuji")
    col_rf, col_xgb = st.columns(2)
    with col_rf:
        st.markdown("#### Random Forest")
        rf_rows = [{"Hyperparameter": k.replace("model__", ""), "Rentang": str(v), "Opsi": len(v)}
               for k, v in RF_PARAM_GRID.items()]
        st.table(pd.DataFrame(rf_rows))
        total_rf = 1
        for v in RF_PARAM_GRID.values():
            total_rf *= len(v)
        st.caption(f"Total kombinasi: **{total_rf}** · Scoring: F1 · CV: Stratified {INNER_SPLITS}-Fold")

    with col_xgb:
        st.markdown("#### XGBoost")
        xgb_rows = [{"Hyperparameter": k.replace("model__", ""), "Rentang": str(v), "Opsi": len(v)}
                for k, v in XGB_PARAM_GRID.items()]
        st.table(pd.DataFrame(xgb_rows))
        total_xgb = 1
        for v in XGB_PARAM_GRID.values():
            total_xgb *= len(v)
        st.caption(f"Total kombinasi: **{total_xgb}** · Scoring: F1 · CV: Stratified {INNER_SPLITS}-Fold")

else:
    # Landing state
    st.info("👈 Isi data nasabah di sidebar lalu klik **Prediksi Sekarang**")

    st.markdown("## ℹ️ Tentang Sistem")
    c1, c2, c3 = st.columns(3)
    for col_w, icon, title, desc in [
        (c1, "🌲", "Random Forest", "Ensemble bagging dari banyak decision tree"),
        (c2, "🚀", "XGBoost", "Gradient boosting yang dioptimasi untuk kecepatan"),
        (c3, "🔒", "Anti-Leakage", "SMOTE & encoding hanya pada training fold"),
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
