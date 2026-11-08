"""
train_models.py
---------------
Trains the five required classifiers on ONE dataset, evaluates them on a
held-out test split, and saves everything the Streamlit app needs:

    model/<model_name>.joblib   -> full sklearn Pipeline (preprocessing + classifier)
    model/meta.json             -> feature columns, target name, class labels
    model/metrics_summary.csv   -> the comparison table (for the README)
    test_data.csv               -> held-out test set (features + target column)

Usage:
    python train_models.py --data my_dataset.csv --target <target_column>
    python train_models.py --data my_dataset.csv --target Class --drop id,Unnamed:0 --seed 7
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# SVC(probability=True) is flagged as deprecated in scikit-learn >= 1.9 (removed in 1.11).
# It still works on the versions we pin in requirements.txt, so keep the output clean.
warnings.filterwarnings("ignore", message=".*probability.*parameter was deprecated.*")

MODEL_DIR = Path("model")

# file name for each model (app.py uses the same names)
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """The six required metrics.
    Binary  -> precision/recall/F1 for the positive class (2nd label).
    Multi   -> weighted average over classes; AUC is one-vs-rest weighted."""
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted",
                labels=list(range(n_classes)),
            )
    except ValueError:  # e.g. a class missing from y_true
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def build_pipelines(num_cols, cat_cols, seed):
    """One shared preprocessing recipe + five different classifiers."""

    def make_preprocessor():
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipe, cat_cols))
        return ColumnTransformer(transformers)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        # var_smoothing=0.1 (default is 1e-9): chosen by 5-fold CV on the TRAINING split only.
        # With many one-hot columns the default makes GaussianNB wildly over-confident.
        "Naive Bayes": GaussianNB(var_smoothing=0.1),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=seed),
    }
    return {
        name: Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        for name, clf in classifiers.items()
    }


def to_markdown(df):
    """Tiny markdown-table writer (avoids needing the 'tabulate' package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the full dataset CSV")
    ap.add_argument("--target", required=True, help="name of the target column")
    ap.add_argument("--drop", default="", help="comma-separated columns to drop (IDs etc.)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    drop_cols = [c.strip() for c in args.drop.split(",") if c.strip()]
    df = df.drop(columns=drop_cols, errors="ignore").dropna(subset=[args.target])

    # ---------- dataset summary (handy for the README) ----------
    feature_cols = [c for c in df.columns if c != args.target]
    print("=" * 70)
    print(f"Rows: {len(df)} | Features: {len(feature_cols)} | Target: {args.target}")
    print("Class distribution:\n", df[args.target].value_counts().to_string())
    print("Missing values (total):", int(df[feature_cols].isna().sum().sum()))
    print("=" * 70)

    num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    # ---------- encode target & split ----------
    classes = sorted(df[args.target].unique().tolist())
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    y = df[args.target].astype(str).map(class_to_idx).astype(int)
    X = df[feature_cols]

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    except ValueError:  # a class too small to stratify
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed)

    # ---------- train + evaluate ----------
    MODEL_DIR.mkdir(exist_ok=True)
    n_classes = len(classes)
    pipelines = build_pipelines(num_cols, cat_cols, args.seed)
    rows = []
    for name, pipe in pipelines.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)
        m = compute_metrics(y_te, pred, proba, n_classes)
        rows.append({"ML Model Name": name, **m})
        joblib.dump(pipe, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"\n--- {name} ---")
        print(confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred, target_names=[str(c) for c in classes], zero_division=0))

    results = pd.DataFrame(rows)
    results.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)

    # ---------- save artifacts for the app ----------
    meta = {
        "target": args.target,
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "classes": [str(c) for c in classes],
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    test_df = X_te.copy()
    test_df[args.target] = [classes[i] for i in y_te]   # original (human-readable) labels
    test_df.to_csv("test_data.csv", index=False)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (paste into README):\n")
    print(to_markdown(results))
    best = results.sort_values("MCC", ascending=False).iloc[0]
    print(f"\nBest by MCC: {best['ML Model Name']} ({best['MCC']:.4f})")
    print(f"\nSaved {len(MODEL_FILES)} models to ./{MODEL_DIR}/ and test_data.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
