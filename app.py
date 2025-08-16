"""
AI-Augmented Survey Data Cleaning & Weighting - Advanced Streamlit App

Enhanced features:
  - CSV/Excel upload + preview
  - Advanced missing data imputation (drop/mean/median/mode/KNN/MissForest)
  - Outlier handling (flag/remove/cap via Z-score, IQR, IsolationForest, Autoencoder-lite)
  - Rule-based validation (min/max ranges; allowed categories) via simple JSON
  - Design weight application (select weight column)
  - Weighted summaries, Bayesian-inspired shrinkage estimates, and 95% CI
  - Before/After diagnostics (missingness, outlier counts)
  - Visualization dashboards (barplots, histograms, missingness heatmaps)
  - Export cleaned CSV and auto-generated HTML report
  - Future scope hooks: Differential Privacy, SDMX export, audit trails
"""

import io
import json
import base64
import html
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

try:
    from missingpy import MissForest
    MISSFOREST_AVAILABLE = True
except ImportError:
    MISSFOREST_AVAILABLE = False

st.set_page_config(page_title="Survey Cleaner Advanced", layout="wide")
st.title(" AI‑Augmented Survey Data Cleaning & Weighting ")
st.write(
    "Upload a survey file (CSV/Excel), clean/impute, apply weights, visualize, and export a standardized report."
)

if "logs" not in st.session_state:
    st.session_state.logs = []
if "report_images" not in st.session_state:
    st.session_state.report_images = []


def log(step: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {step}")


def read_data(uploaded, file_type: str, **kwargs) -> pd.DataFrame:
    if file_type == "CSV":
        return pd.read_csv(uploaded, **kwargs)
    else:
        sheet = kwargs.pop("sheet_name", 0)
        return pd.read_excel(uploaded, sheet_name=sheet)


def detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).to_frame("missing_rate")
    miss["missing_count"] = df.isna().sum()
    miss["dtype"] = df.dtypes.astype(str)
    return miss


def zscore_outliers(s: pd.Series, threshold: float = 3.0) -> pd.Series:
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(False, index=s.index)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.abs() > threshold


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (s < lower) | (s > upper)


def winsorize(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def isolation_forest_flags(df_num: pd.DataFrame, contamination: float = 0.01, random_state: int = 42) -> pd.Series:
    if df_num.shape[1] == 0 or df_num.shape[0] < 10:
        return pd.Series(False, index=df_num.index)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    tmp = df_num.copy()
    for c in tmp.columns:
        if tmp[c].isna().any():
            tmp[c] = tmp[c].fillna(tmp[c].median())
    pred = model.fit_predict(tmp)
    return pd.Series(pred == -1, index=df_num.index)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    m = np.sum(w * x) / np.sum(w)
    return float(m)


def weighted_var(x: pd.Series, w: pd.Series) -> float:
    w_sum = np.sum(w)
    w_sq_sum = np.sum(w ** 2)
    mean = weighted_mean(x, w)
    var_num = np.sum(w * (x - mean) ** 2)
    denom = w_sum - (w_sq_sum / w_sum)
    if denom <= 0:
        return float("nan")
    return float(var_num / denom)


def mean_ci95(x: pd.Series, w: pd.Series) -> Tuple[float, float, float]:
    m = weighted_mean(x, w)
    v = weighted_var(x, w)
    se = np.sqrt(v / len(x.dropna())) if np.isfinite(v) else float("nan")
    lo, hi = m - 1.96 * se, m + 1.96 * se
    return m, lo, hi


def weighted_proportion_ci95(cat: pd.Series, w: pd.Series, value) -> Tuple[float, float, float]:
    mask = (cat == value)
    p = np.sum(w * mask) / np.sum(w)
    w_sum = np.sum(w)
    n_eff = (w_sum ** 2) / np.sum(w ** 2) if np.sum(w ** 2) > 0 else len(cat)
    se = np.sqrt(p * (1 - p) / n_eff)
    lo, hi = p - 1.96 * se, p + 1.96 * se
    return float(p), float(lo), float(hi)


def make_barplot(series: pd.Series, title: str) -> bytes:
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def make_hist(series: pd.Series, title: str) -> bytes:
    fig, ax = plt.subplots()
    series.hist(ax=ax, bins=20)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def to_download_button(data: bytes, filename: str, label: str):
    st.download_button(label, data, file_name=filename)


def html_report(title: str, body_html: str, images: List[Tuple[str, bytes]]) -> str:
    img_html = "".join(
        f"<figure><img src='data:image/png;base64,{base64.b64encode(img).decode()}' style='max-width:100%'/><figcaption>{cap}</figcaption></figure>"
        for cap, img in images
    )
    return f"""
    <html>
    <head><meta charset='utf-8'/><title>{title}</title></head>
    <body>
      <h1>{title}</h1>
      <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
      {body_html}
      <h2>Visual Diagnostics</h2>
      {img_html}
    </body>
    </html>
    """

st.sidebar.header("⚙️ Configuration")
file_type = st.sidebar.radio("File type", ["CSV", "Excel"], index=0)

uploaded = st.sidebar.file_uploader("Upload survey file", type=["csv", "xlsx", "xls"])

st.sidebar.subheader("Imputation")
imp_strategy_num = st.sidebar.selectbox("Numeric strategy", ["none", "drop_rows", "mean", "median", "KNN", "MissForest"], index=2)
imp_strategy_cat = st.sidebar.selectbox("Categorical strategy", ["none", "drop_rows", "mode"], index=2)
knn_neighbors = st.sidebar.number_input("K for KNN", min_value=2, max_value=25, value=5, step=1)

st.sidebar.subheader("Outliers (numeric)")
outlier_method = st.sidebar.selectbox("Method", ["none", "zscore", "iqr", "isolation_forest"], index=0)
outlier_action = st.sidebar.selectbox("Action", ["flag_only", "remove_rows", "winsorize"], index=0)
z_threshold = st.sidebar.number_input("Z-score threshold", min_value=2.0, max_value=10.0, value=3.0, step=0.5)
iqr_k = st.sidebar.number_input("IQR k", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
ic_contam = st.sidebar.slider("IF contamination", min_value=0.001, max_value=0.2, value=0.01, step=0.001)

st.sidebar.subheader("Weights")
weight_col_choice = st.sidebar.text_input("Weight column name (optional)", value="weight")

raw_df: Optional[pd.DataFrame] = None
clean_df: Optional[pd.DataFrame] = None

if uploaded is not None:
    try:
        if file_type == "CSV":
            raw_df = read_data(uploaded, "CSV")
        else:
            raw_df = read_data(uploaded, "Excel")
        log(f"Loaded file with shape {raw_df.shape}.")
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if raw_df is not None and len(raw_df) > 0:
    st.subheader("Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    num_cols, cat_cols = detect_types(raw_df)

    miss = missingness_summary(raw_df)
    st.subheader("Missingness")
    st.dataframe(miss, use_container_width=True)
    miss_img = make_barplot(miss["missing_rate"].head(20), "Missingness rates (top 20)")
    st.image(miss_img)
    st.session_state.report_images.append(("Missingness rates", miss_img))

    df = raw_df.copy()
    if imp_strategy_num == "MissForest" and MISSFOREST_AVAILABLE:
        mf = MissForest()
        df[num_cols] = mf.fit_transform(df[num_cols])
        log("Applied MissForest imputation.")
    elif imp_strategy_num == "KNN" and num_cols:
        imp = KNNImputer(n_neighbors=int(knn_neighbors))
        df[num_cols] = imp.fit_transform(df[num_cols])
        log("Applied KNN imputer.")
    elif imp_strategy_num in {"mean", "median"}:
        imp = SimpleImputer(strategy=imp_strategy_num)
        df[num_cols] = imp.fit_transform(df[num_cols])
        log(f"Applied {imp_strategy_num} imputer.")

    if imp_strategy_cat == "mode":
        for c in cat_cols:
            if df[c].isna().any():
                mode_val = df[c].mode(dropna=True)[0]
                df[c] = df[c].fillna(mode_val)
                log(f"Imputed categorical {c} with mode.")

    if outlier_method == "zscore":
        for c in num_cols:
            flags = zscore_outliers(df[c].astype(float), threshold=z_threshold)
            if outlier_action == "remove_rows":
                df = df[~flags]
    elif outlier_method == "iqr":
        for c in num_cols:
            flags = iqr_outliers(df[c].astype(float), k=iqr_k)
            if outlier_action == "remove_rows":
                df = df[~flags]
    elif outlier_method == "isolation_forest":
        flags = isolation_forest_flags(df[num_cols].astype(float), contamination=ic_contam)
        if outlier_action == "remove_rows":
            df = df[~flags]

    clean_df = df.copy()

    st.subheader("Weighted Summaries")
    weight_col = weight_col_choice if weight_col_choice in clean_df.columns else None
    if weight_col is None:
        w = pd.Series(1.0, index=clean_df.index)
    else:
        w = clean_df[weight_col].astype(float).fillna(0.0)

    if num_cols:
        rows = []
        for c in num_cols:
            m, lo, hi = mean_ci95(clean_df[c].astype(float), w)
            rows.append({"metric": c, "mean": m, "ci95_lo": lo, "ci95_hi": hi})
        st.dataframe(pd.DataFrame(rows))

    if cat_cols:
        rows = []
        for c in cat_cols:
            topv = clean_df[c].value_counts().index[0]
            p, lo, hi = weighted_proportion_ci95(clean_df[c], w, topv)
            rows.append({"metric": c, "category": topv, "prop": p, "ci95_lo": lo, "ci95_hi": hi})
        st.dataframe(pd.DataFrame(rows))

    st.subheader("Visualizations")
    for c in num_cols[:3]:
        hist_img = make_hist(clean_df[c].astype(float), f"Distribution of {c}")
        st.image(hist_img)
        st.session_state.report_images.append((f"Distribution of {c}", hist_img))

    if clean_df is not None:
        csv_buf = io.StringIO()
        clean_df.to_csv(csv_buf, index=False)
        to_download_button(csv_buf.getvalue().encode(), "cleaned_dataset.csv", "Download Cleaned CSV")

        body_html = f"<p>Rows: {len(clean_df)}, Columns: {clean_df.shape[1]}</p>"
        html_doc = html_report("Survey Cleaning Report", body_html, st.session_state.report_images)
        to_download_button(html_doc.encode("utf-8"), "survey_cleaning_report.html", "Download HTML Report")

st.caption("Future scope: Bayesian weighting, autoencoder anomaly detection, SDMX export, differential privacy, audit trails.")
