"""
app.py  -  Streamlit front-end for the 5-model classification assignment.

The app NEVER retrains anything: it loads the saved pipelines from ./model,
scores the uploaded test CSV, and shows metrics / confusion matrix / report.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Shopper Intent Predictor", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return meta, models


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    avg = "binary" if n_classes == 2 else "weighted"
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                                labels=list(range(n_classes)))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def split_features_target(df, meta):
    """Validate the uploaded CSV and return (X, y_encoded). Stops the app with a
    friendly message if columns / labels do not match what the models expect."""
    needed = meta["feature_columns"] + [meta["target"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing required column(s): {missing}")
        st.stop()
    df = df.dropna(subset=[meta["target"]])
    mapping = {c: i for i, c in enumerate(meta["classes"])}
    y = df[meta["target"]].astype(str).map(mapping)
    if y.isna().any():
        bad = sorted(df.loc[y.isna(), meta["target"]].astype(str).unique())
        st.error(f"Target column contains labels the models were not trained on: {bad}")
        st.stop()
    return df[meta["feature_columns"]], y.astype(int), df


def plot_confusion(cm, class_names):
    size = max(4, 0.6 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
st.title("🛒 Shopper Intent Predictor")
st.caption("Will an online shopping session end in a purchase? Choose one of five trained "
           "classifiers, upload the test CSV, and compare how each one performs.")

try:
    meta, models = load_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the saved models from /model: {exc}")
    st.stop()

class_names = meta["classes"]
n_classes = len(class_names)

with st.sidebar:
    st.header("🎛️ Model & data")
    model_name = st.selectbox("Select a classifier", list(models.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = False
    if SAMPLE_CSV.exists():
        use_sample = st.checkbox("…or use the bundled test_data.csv", value=False)
    st.markdown("---")
    st.markdown(f"**Target column:** `{meta['target']}`")
    st.markdown(f"**Classes:** {', '.join(class_names)}")
    st.markdown(f"**Features expected:** {len(meta['feature_columns'])}")
    with st.expander("Show required column names"):
        st.write(meta["feature_columns"] + [meta["target"]])

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
elif use_sample:
    raw_df = pd.read_csv(SAMPLE_CSV)
else:
    st.info("👈 Upload a CSV in the sidebar (it must contain every feature column and the "
            "target column) or tick the bundled-sample checkbox.")
    st.stop()

X, y, clean_df = split_features_target(raw_df, meta)

# score every model once (cheap on test-sized data)
results, preds, probas = {}, {}, {}
for name, pipe in models.items():
    preds[name] = pipe.predict(X)
    probas[name] = pipe.predict_proba(X)
    results[name] = compute_metrics(y, preds[name], probas[name], n_classes)
results_df = pd.DataFrame(results).T.round(4)

tab_one, tab_all, tab_data, tab_about = st.tabs(
    [f"📈 {model_name}", "🏁 Model comparison", "🔍 Data & predictions", "ℹ️ About"])

with tab_one:
    st.subheader(f"Evaluation metrics - {model_name}")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, results[model_name].items()):
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, preds[model_name], labels=list(range(n_classes)))
        st.pyplot(plot_confusion(cm, class_names))
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, preds[model_name], labels=list(range(n_classes)),
                                    target_names=class_names, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")
   
	# ---- ROC curve (binary problems only) ----
    if n_classes == 2:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, probas[model_name][:, 1])
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {results[model_name]['AUC']:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="grey")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.legend(loc="lower right")
        st.subheader("ROC curve")
        st.pyplot(fig_roc)

with tab_all:
    st.subheader("All five models on the uploaded data")
    st.dataframe(results_df, width="stretch")
    best = results_df["MCC"].idxmax()
    st.success(f"Highest MCC on this data: **{best}** ({results_df.loc[best, 'MCC']:.4f})")
    st.bar_chart(results_df[["Accuracy", "F1", "MCC"]])
    ref = MODEL_DIR / "metrics_summary.csv"
    if ref.exists():
        with st.expander("Metrics recorded at training time (README table)"):
            st.dataframe(pd.read_csv(ref).round(4), width="stretch", hide_index=True)

with tab_data:
    st.subheader("Uploaded data preview")
    st.write(f"{len(clean_df)} rows x {clean_df.shape[1]} columns")
    st.dataframe(clean_df.head(50), width="stretch")

    out = clean_df.copy()
    out["predicted_" + meta["target"]] = [class_names[i] for i in preds[model_name]]
    out["confidence"] = probas[model_name].max(axis=1).round(4)
    st.subheader(f"Predictions from {model_name}")
    st.dataframe(out.head(50), width="stretch")
    st.download_button("⬇️ Download all predictions (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")
                       
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
- **Goal:** predict whether a website session ends in a purchase (`Revenue` = 1).
- **Data:** UCI *Online Shoppers Purchasing Intention* dataset (Sakar & Kastro, 2018),
  de-duplicated and extended with five engineered features.
- **Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM.
- **Why not just accuracy?** Only about 15.6% of sessions are purchases, so MCC, F1 and
  recall say more about model quality than accuracy.
        """
    )
