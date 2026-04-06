"""Random Forest training script extracted from the notebook workflow.

This script keeps the modeling logic notebook-compatible while using
leakage-safe preprocessing per fold:
- median/mode imputation is fit on train fold only
- ordinal encoding is fit on train fold only
- SMOTE is applied on train fold only
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/loan_data.csv")
TARGET_COL = "status_pinjaman"
DROP_COLS = ["id_pelanggan", "gagal_bayar_tercatat"]
N_SPLITS = 10
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


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

        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X_train_proc[cat_cols] = cat_encoder.fit_transform(X_train_cat)
        X_val_proc[cat_cols] = cat_encoder.transform(X_val_cat)

    return X_train_proc.astype(float), X_val_proc.astype(float)


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
    summary = metrics_df[["accuracy", "precision", "recall", "f1_score", "auc"]].agg(["mean", "std"])
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


def print_feature_importance(X: pd.DataFrame, avg_importance: np.ndarray | None, top_n: int = 10) -> None:
    if avg_importance is None:
        print("\nFeature importance is not available for this model.")
        return

    fi_df = (
        pd.DataFrame({"feature": X.columns, "importance": avg_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["rank"] = np.arange(1, len(fi_df) + 1)

    print(f"\nTop {top_n} feature importance:")
    print(fi_df.head(top_n).to_string(index=False, float_format=lambda v: f"{v:.6f}"))


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

    print("\nRandom Forest hyperparameters:")
    for key, value in RF_PARAMS.items():
        print(f"- {key}: {value}")

    base_model = RandomForestClassifier(**RF_PARAMS)
    result = evaluate_model_cv(X=X, y=y, model=base_model)

    print_cv_result(result)
    print_best_fold_analysis(result)
    print_feature_importance(X, result["avg_importance"], top_n=10)
    run_label_shuffle_sanity_test(X, y, base_model)


if __name__ == "__main__":
    main()
