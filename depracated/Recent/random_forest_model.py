"""Random Forest training script extracted from the notebook workflow.

This script keeps the modeling logic notebook-compatible while using
leakage-safe preprocessing per fold:
- median/mode imputation is fit on train fold only
- One-Hot Encoding is fit on train fold only (no ordinal bias)
- SMOTE is applied on train fold only
- F2-Score and financial loss estimation included
- GridSearchCV hyperparameter tuning with detailed search table
"""

from pathlib import Path
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/loan_data.csv")
TARGET_COL = "status_pinjaman"
DROP_COLS = ["id_pelanggan", "gagal_bayar_tercatat"]
N_SPLITS = 10
INNER_SPLITS = 3
RANDOM_STATE = 42

# --- Hyperparameter Search Space ---
RF_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [10, 15],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

# Fallback defaults (used if GridSearchCV is skipped)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Financial loss constants (customize to your domain)
AVG_LOAN_AMOUNT = 33_042  # Mean jumlah_pinjaman from the dataset
RECOVERY_RATE = 0.30      # Assumed 30% recovery on defaulted loans
PROFIT_MARGIN = 0.05      # Profit margin per good loan (missed if FN)


def load_features_target(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df_raw = pd.read_csv(dataset_path)
    missing_drop_cols = [c for c in DROP_COLS if c not in df_raw.columns]
    if missing_drop_cols:
        raise KeyError(f"Drop columns not found in dataset: {missing_drop_cols}")

    if TARGET_COL not in df_raw.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")

    df_model = df_raw.drop(columns=DROP_COLS)
    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    print("=" * 70)
    print("RANDOM FOREST MODEL")
    print("=" * 70)
    print(f"Rows                : {len(df_raw):,}")
    print(f"Columns (raw)       : {df_raw.shape[1]}")
    print(f"Columns (modeling)  : {df_model.shape[1]}")
    print(f"Total features      : {X.shape[1]}")
    print(f"Class distribution  :")
    print(y.value_counts().sort_index().to_string())

    return X, y


def preprocess_fold_no_leakage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess a single fold without data leakage.

    Uses One-Hot Encoding instead of Ordinal Encoding for nominal
    categorical variables (status_pekerjaan, tipe_produk, tujuan_pinjaman)
    to prevent the algorithm from learning spurious ordinal relationships.
    """
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train_proc[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_val_proc[num_cols] = num_imputer.transform(X_val[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
        X_val_cat = cat_imputer.transform(X_val[cat_cols])

        # One-Hot Encoding: no ordinal bias for nominal variables
        cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        X_train_ohe = cat_encoder.fit_transform(X_train_cat)
        X_val_ohe = cat_encoder.transform(X_val_cat)

        ohe_cols = cat_encoder.get_feature_names_out(cat_cols)
        X_train_ohe_df = pd.DataFrame(
            X_train_ohe, columns=ohe_cols, index=X_train.index
        )
        X_val_ohe_df = pd.DataFrame(
            X_val_ohe, columns=ohe_cols, index=X_val.index
        )

        X_train_proc = X_train_proc.drop(columns=cat_cols)
        X_val_proc = X_val_proc.drop(columns=cat_cols)
        X_train_proc = pd.concat([X_train_proc, X_train_ohe_df], axis=1)
        X_val_proc = pd.concat([X_val_proc, X_val_ohe_df], axis=1)

    return X_train_proc.astype(float), X_val_proc.astype(float)


def get_representative_params(best_params_per_fold: list[dict]) -> dict:
    if not best_params_per_fold:
        return {"random_state": RANDOM_STATE, "n_jobs": -1}
    params_repr = {}
    for key in best_params_per_fold[0]:
        vals = [p[key] for p in best_params_per_fold]
        params_repr[key] = Counter(vals).most_common(1)[0][0]
    params_repr["random_state"] = RANDOM_STATE
    params_repr["n_jobs"] = -1
    return params_repr


def evaluate_model_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    inner_splits: int = INNER_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    rows: list[dict] = []
    oof_y_true: list[int] = []
    oof_y_pred: list[int] = []
    oof_y_prob: list[float] = []
    best_params_per_fold: list[dict] = []
    all_importances = []
    feature_names = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        X_train_proc, X_val_proc = preprocess_fold_no_leakage(
            X_train, X_val, num_cols=num_cols, cat_cols=cat_cols
        )

        if feature_names is None:
            feature_names = X_train_proc.columns.tolist()

        pipeline = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=random_state)),
                ("model", RandomForestClassifier(random_state=random_state, n_jobs=1)),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            RF_PARAM_GRID,
            scoring="f1",
            cv=inner_cv,
            n_jobs=-1,
        )
        grid.fit(X_train_proc, y_train)

        best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        best_params_per_fold.append(best_params)

        smote_fold = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote_fold.fit_resample(X_train_proc, y_train)

        fold_model = RandomForestClassifier(
            **best_params,
            random_state=random_state,
            n_jobs=-1,
        )
        fold_model.fit(X_train_smote, y_train_smote)

        y_pred = fold_model.predict(X_val_proc)
        y_prob = fold_model.predict_proba(X_val_proc)[:, 1]

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
            "f2_score": fbeta_score(y_val, y_pred, beta=2, zero_division=0),
            "auc": roc_auc_score(y_val, y_prob),
        })

        oof_y_true.extend(y_val.tolist())
        oof_y_pred.extend(y_pred.tolist())
        oof_y_prob.extend(y_prob.tolist())

        if hasattr(fold_model, "feature_importances_"):
            all_importances.append(fold_model.feature_importances_)

    metrics_df = pd.DataFrame(rows)
    summary = metrics_df[
        ["accuracy", "precision", "recall", "f1_score", "f2_score", "auc"]
    ].agg(["mean", "std"])
    avg_importance = np.mean(all_importances, axis=0) if all_importances else None

    return {
        "metrics_df": metrics_df,
        "summary": summary,
        "best_params_per_fold": best_params_per_fold,
        "oof_y_true": oof_y_true,
        "oof_y_pred": oof_y_pred,
        "oof_y_prob": oof_y_prob,
        "avg_importance": avg_importance,
        "feature_names": feature_names,
    }


def evaluate_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: RandomForestClassifier,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    use_smote: bool = True,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    smote = SMOTE(random_state=random_state)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    rows: list[dict] = []
    best = {
        "f1_score": -1.0,
        "fold": None,
        "y_true": None,
        "y_pred": None,
        "y_prob": None,
    }
    all_importances = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        X_train_proc, X_val_proc = preprocess_fold_no_leakage(
            X_train, X_val, num_cols=num_cols, cat_cols=cat_cols
        )

        if use_smote:
            X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)

        fold_model = clone(model)
        fold_model.fit(X_train_proc, y_train)

        y_pred = fold_model.predict(X_val_proc)
        y_prob = fold_model.predict_proba(X_val_proc)[:, 1]

        row = {
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "f2_score": fbeta_score(y_val, y_pred, beta=2, zero_division=0),
            "auc": roc_auc_score(y_val, y_prob),
        }
        rows.append(row)

        if hasattr(fold_model, "feature_importances_"):
            all_importances.append(fold_model.feature_importances_)

        if row["f1_score"] > best["f1_score"]:
            best = {
                "f1_score": row["f1_score"],
                "fold": fold,
                "y_true": y_val,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

    metrics_df = pd.DataFrame(rows)
    summary = metrics_df[
        ["accuracy", "precision", "recall", "f1_score", "f2_score", "auc"]
    ].agg(["mean", "std"])
    avg_importance = np.mean(all_importances, axis=0) if all_importances else None

    return {
        "metrics_df": metrics_df,
        "summary": summary,
        "best": best,
        "avg_importance": avg_importance,
    }


def print_cv_result(result: dict) -> None:
    metrics_df = result["metrics_df"]
    summary = result["summary"]

    print("\nPer-fold metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSummary (mean/std):")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


def print_fold_metrics_table(metrics_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"{'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'F2-Score':>10} {'AUC':>10}")
    print("=" * 70)

    for _, row in metrics_df.iterrows():
        print(
            f"  {int(row['fold']):<4} {row['accuracy']:>10.4f} {row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} {row['f1_score']:>10.4f} {row['f2_score']:>10.4f} {row['auc']:>10.4f}"
        )

    mean_vals = metrics_df[["accuracy", "precision", "recall", "f1_score", "f2_score", "auc"]].mean()
    std_vals = metrics_df[["accuracy", "precision", "recall", "f1_score", "f2_score", "auc"]].std()

    print("=" * 70)
    print(
        f"  {'Mean':<4} {mean_vals['accuracy']:>10.4f} {mean_vals['precision']:>10.4f} "
        f"{mean_vals['recall']:>10.4f} {mean_vals['f1_score']:>10.4f} {mean_vals['f2_score']:>10.4f} {mean_vals['auc']:>10.4f}"
    )
    print(
        f"  {'Std':<4} {std_vals['accuracy']:>10.4f} {std_vals['precision']:>10.4f} "
        f"{std_vals['recall']:>10.4f} {std_vals['f1_score']:>10.4f} {std_vals['f2_score']:>10.4f} {std_vals['auc']:>10.4f}"
    )
    print("=" * 70)


def print_best_fold_analysis(result: dict) -> None:
    best = result["best"]

    print("\nBest fold analysis")
    print("-" * 70)
    print(f"Best fold (by F1) : {best['fold']}")
    print(f"Best fold F1       : {best['f1_score']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            best["y_true"],
            best["y_pred"],
            target_names=["Gagal Bayar (0)", "Lunas (1)"],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(best["y_true"], best["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (best fold):")
    print(cm)
    print(f"TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    # --- F2-Score ---
    f2 = fbeta_score(best["y_true"], best["y_pred"], beta=2, zero_division=0)
    print(f"\nF2-Score (best fold): {f2:.4f}")
    print("  (F2 memberi bobot lebih pada recall → lebih sensitif mendeteksi gagal bayar)")

    # --- Financial loss estimation ---
    print("\nEstimasi Kerugian Finansial (best fold):")
    print("-" * 50)
    loss_per_fp = AVG_LOAN_AMOUNT * (1 - RECOVERY_RATE)
    cost_per_fn = AVG_LOAN_AMOUNT * PROFIT_MARGIN
    total_fp_loss = fp * loss_per_fp
    total_fn_cost = fn * cost_per_fn
    total_cost = total_fp_loss + total_fn_cost

    print(f"  Rata-rata jumlah pinjaman       : Rp {AVG_LOAN_AMOUNT:,.0f}")
    print(f"  Recovery rate                   : {RECOVERY_RATE:.0%}")
    print(f"  Kerugian per FP (gagal bayar)   : Rp {loss_per_fp:,.0f}")
    print(f"  Opportunity cost per FN         : Rp {cost_per_fn:,.0f}")
    print(f"  Total FP ({fp:,} kasus)            : Rp {total_fp_loss:,.0f}")
    print(f"  Total FN ({fn:,} kasus)            : Rp {total_fn_cost:,.0f}")
    print(f"  Total estimasi kerugian         : Rp {total_cost:,.0f}")


def print_feature_importance(feature_names: list[str], avg_importance: np.ndarray | None, top_n: int = 10) -> None:
    if avg_importance is None or not feature_names:
        print("\nFeature importance is not available for this model.")
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": avg_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["rank"] = np.arange(1, len(fi_df) + 1)

    print(f"\nTop {top_n} feature importance:")
    print(fi_df.head(top_n).to_string(index=False, float_format=lambda v: f"{v:.6f}"))


def print_hyperparameter_search_table() -> None:
    """Print a detailed table of hyperparameter ranges explored."""
    print("\n" + "=" * 70)
    print("TABEL RENTANG HYPERPARAMETER YANG DIUJI (Random Forest)")
    print("=" * 70)

    table_data = []
    for param, values in RF_PARAM_GRID.items():
        clean_name = param.replace("model__", "")
        table_data.append({
            "Hyperparameter": clean_name,
            "Rentang Nilai": str(values),
            "Jumlah Opsi": len(values),
        })

    df_table = pd.DataFrame(table_data)
    print(df_table.to_string(index=False))

    total_combinations = 1
    for v in RF_PARAM_GRID.values():
        total_combinations *= len(v)
    print(f"\nTotal kombinasi yang diuji : {total_combinations}")
    print(f"Metode pencarian           : GridSearchCV (exhaustive)")
    print(f"Scoring                    : F1-Score")
    print(f"Cross-validation           : Stratified {INNER_SPLITS}-Fold")


def run_hyperparameter_tuning(X: pd.DataFrame, y: pd.Series) -> dict:
    """Run GridSearchCV to find the best hyperparameters."""
    print("\nRunning GridSearchCV for Random Forest...")

    # One-Hot Encode first (for GridSearchCV)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_proc = X.copy()
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_proc[num_cols] = num_imputer.fit_transform(X[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_proc[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=False)

    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )

    grid_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipeline,
        RF_PARAM_GRID,
        scoring="f1",
        cv=grid_cv,
        n_jobs=-1,
    )

    grid.fit(X_proc, y)

    best_params = {
        k.replace("model__", ""): v for k, v in grid.best_params_.items()
    }
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1

    print(f"Best Params RF : {best_params}")
    print(f"Best CV F1 RF  : {grid.best_score_:.4f}")

    return best_params


def run_label_shuffle_sanity_test(X: pd.DataFrame, y: pd.Series, model: RandomForestClassifier) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)

    test_result = evaluate_model_cv(
        X=X,
        y=y_shuffled,
        model=model,
        n_splits=5,
        random_state=RANDOM_STATE,
        use_smote=False,
    )
    auc_mean = test_result["summary"].loc["mean", "auc"]
    auc_std = test_result["summary"].loc["std", "auc"]

    print("\nLabel shuffle sanity test (expected AUC near 0.5):")
    print(f"Shuffled AUC mean/std : {auc_mean:.4f} +/- {auc_std:.4f}")


def main() -> None:
    X, y = load_features_target(DATA_PATH)

    # Print hyperparameter search table
    print_hyperparameter_search_table()

    # Nested CV evaluation (GridSearch inside each fold)
    result = evaluate_model_nested_cv(X=X, y=y)
    print_fold_metrics_table(result["metrics_df"])

    print("\nHyperparameter RF terpilih per fold:")
    for i, p in enumerate(result["best_params_per_fold"], 1):
        print(f"  Fold {i:2d}: {p}")

    rf_params_repr = get_representative_params(result["best_params_per_fold"])
    print(f"\nHyperparameter representatif (modus): {rf_params_repr}")

    # OOF classification report
    print("\nClassification Report - Random Forest (Out-of-Fold):")
    print(
        classification_report(
            result["oof_y_true"],
            result["oof_y_pred"],
            target_names=["Gagal Bayar (0)", "Lunas (1)"],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(result["oof_y_true"], result["oof_y_pred"])
    tn, fp, fn, tp = cm.ravel()

    print("\nF2-Score (OOF):")
    f2_oof = fbeta_score(result["oof_y_true"], result["oof_y_pred"], beta=2, zero_division=0)
    print(f"  F2-Score: {f2_oof:.4f}")

    print("\nEstimasi Kerugian Finansial (OOF):")
    print("-" * 50)
    loss_per_fp = AVG_LOAN_AMOUNT * (1 - RECOVERY_RATE)
    cost_per_fn = AVG_LOAN_AMOUNT * PROFIT_MARGIN
    total_fp_loss = fp * loss_per_fp
    total_fn_cost = fn * cost_per_fn
    total_cost = total_fp_loss + total_fn_cost

    print(f"  Rata-rata jumlah pinjaman       : Rp {AVG_LOAN_AMOUNT:,.0f}")
    print(f"  Recovery rate                   : {RECOVERY_RATE:.0%}")
    print(f"  Kerugian per FP (gagal bayar)   : Rp {loss_per_fp:,.0f}")
    print(f"  Opportunity cost per FN         : Rp {cost_per_fn:,.0f}")
    print(f"  Total FP ({fp:,} kasus)            : Rp {total_fp_loss:,.0f}")
    print(f"  Total FN ({fn:,} kasus)            : Rp {total_fn_cost:,.0f}")
    print(f"  Total estimasi kerugian         : Rp {total_cost:,.0f}")

    print_feature_importance(result["feature_names"], result["avg_importance"], top_n=10)

    base_model = RandomForestClassifier(**rf_params_repr)
    run_label_shuffle_sanity_test(X, y, base_model)


if __name__ == "__main__":
    main()
