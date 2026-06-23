from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.core import (
    read_table,
    train_models,
    predict_dataframe,
    feature_signal_report,
    find_target_column,
)

st.set_page_config(
    page_title="SmartEvent AI",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(37,99,235,.18), transparent 27%),
        radial-gradient(circle at 90% 10%, rgba(34,197,94,.10), transparent 24%),
        linear-gradient(135deg, #050b14, #091526);
}
[data-testid="stSidebar"] {
    background: #07111f;
    border-right: 1px solid rgba(148,163,184,.22);
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
h1 {font-size: 2.35rem !important; letter-spacing: -.035em;}
.small {color: #94a3b8; font-size: .95rem;}
.kpi {
    border: 1px solid rgba(148,163,184,.22);
    background: linear-gradient(180deg, rgba(17,31,51,.98), rgba(9,19,33,.98));
    border-radius: 18px;
    padding: 18px 20px;
    min-height: 118px;
    box-shadow: 0 16px 42px rgba(0,0,0,.18);
}
.kpi-label {color: #94a3b8; font-size: .86rem; margin-bottom: 8px;}
.kpi-value {font-size: 2rem; font-weight: 850; letter-spacing: -.03em;}
.kpi-sub {color: #94a3b8; margin-top: 8px; font-size: .84rem;}
.sec {
    font-size: 1.35rem;
    font-weight: 850;
    margin-top: 1.35rem;
    margin-bottom: .75rem;
}
.card {
    border: 1px solid rgba(148,163,184,.22);
    background: rgba(15,27,45,.94);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 14px 38px rgba(0,0,0,.16);
}
.info-box {
    background:rgba(96,165,250,.12);
    border:1px solid rgba(96,165,250,.32);
    border-left:4px solid #60a5fa;
    border-radius:14px;
    padding:13px 16px;
    margin:16px 0 10px;
}
.warn-box {
    background:rgba(248,113,113,.13);
    border:1px solid rgba(248,113,113,.35);
    border-left:4px solid #f87171;
    border-radius:14px;
    padding:13px 16px;
    margin:16px 0 10px;
}
</style>
""", unsafe_allow_html=True)


def pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def metric(artifact, name):
    table = artifact["metrics"]
    row = table[table["Model"] == artifact["best_model"]]
    if row.empty:
        return None
    return float(row.iloc[0][name])


with st.sidebar:
    st.markdown("## 📅 SmartEvent AI")
    st.markdown("<span class='small'>Upload-only attendance prediction dashboard</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Required upload")
    st.markdown(
        """
        Upload CSV / Excel with target column:

        - `Attended` or `attendance`
        - Values: `Yes/No` or `1/0`
        """
    )
    st.markdown("---")
    st.markdown("### Strong features")
    st.markdown(
        """
        Better accuracy needs:
        - `Distance_KM`
        - `Reminder_Clicked`
        - `Email_Opened`
        - `SMS_Confirmed`
        - `Previous_No_Show_Count`
        - `Attendance_Rate`
        """
    )

st.title("SmartEvent AI Attendance Prediction")
st.markdown("<span class='small'>Open the application first, then upload your CSV/Excel file. No preinstalled dataset is required.</span>", unsafe_allow_html=True)

st.markdown("<div class='sec'>1) Upload Dataset</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload registration data", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.markdown(
        """
        <div class='info-box'>
        <b>Waiting for upload.</b><br>
        This application does not require a built-in dataset. Upload your CSV or Excel file to train the model and generate predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()

try:
    df = read_table(uploaded)
except Exception as exc:
    st.error(f"File reading failed: {exc}")
    st.stop()

target = find_target_column(df)

if target is None:
    st.markdown(
        """
        <div class='warn-box'>
        <b>Target column missing.</b><br>
        Your file must contain <b>Attended</b> or <b>attendance</b> column with Yes/No or 1/0 values.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(20), use_container_width=True)
    st.stop()

st.success(f"File loaded successfully: {len(df):,} rows, {len(df.columns):,} columns. Target column: {target}")

with st.spinner("Training models and generating predictions..."):
    artifact = train_models(df)
    predictions = predict_dataframe(df, artifact)

accuracy = metric(artifact, "Accuracy")
f1 = metric(artifact, "F1 Score")
roc = metric(artifact, "ROC-AUC")

st.markdown("<div class='sec'>2) Model Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>Best model</div><div class='kpi-value'>{artifact['best_model']}</div><div class='kpi-sub'>{len(artifact['feature_columns'])} features used</div></div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>Accuracy</div><div class='kpi-value' style='color:#22c55e'>{pct(accuracy)}</div><div class='kpi-sub'>Correct predictions</div></div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>F1 Score</div><div class='kpi-value' style='color:#a78bfa'>{pct(f1)}</div><div class='kpi-sub'>Balanced performance</div></div>",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>ROC-AUC</div><div class='kpi-value' style='color:#60a5fa'>{pct(roc)}</div><div class='kpi-sub'>Ranking quality</div></div>",
        unsafe_allow_html=True,
    )

if roc is not None and roc < 55:
    st.markdown(
        """
        <div class='warn-box'>
        <b>Why accuracy is low:</b><br>
        Dataset has weak correlation with attendance. Add distance, reminder response, email open, SMS confirmation,
        previous no-show history, and attendance rate.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='sec'>3) Model Performance Comparison</div>", unsafe_allow_html=True)
st.dataframe(
    artifact["metrics"],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.2f%%"),
        "F1 Score": st.column_config.ProgressColumn("F1 Score", min_value=0, max_value=100, format="%.2f%%"),
        "ROC-AUC": st.column_config.ProgressColumn("ROC-AUC", min_value=0, max_value=100, format="%.2f%%"),
    },
)

st.markdown("<div class='sec'>4) Prediction Summary</div>", unsafe_allow_html=True)

total = len(predictions)
attend_count = int((predictions["prediction"] == "Attend").sum())
noshow_count = total - attend_count
avg_probability = float(predictions["attendance_probability"].mean())

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Total registrations</div><div class='kpi-value'>{total:,}</div><div class='kpi-sub'>Uploaded records</div></div>", unsafe_allow_html=True)
with s2:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Predicted attend</div><div class='kpi-value' style='color:#22c55e'>{attend_count:,}</div><div class='kpi-sub'>{attend_count / total * 100:.2f}%</div></div>", unsafe_allow_html=True)
with s3:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Predicted no-show</div><div class='kpi-value' style='color:#f87171'>{noshow_count:,}</div><div class='kpi-sub'>{noshow_count / total * 100:.2f}%</div></div>", unsafe_allow_html=True)
with s4:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Average probability</div><div class='kpi-value' style='color:#60a5fa'>{avg_probability:.2f}%</div><div class='kpi-sub'>Mean attend chance</div></div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>5) Charts</div>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Predicted Attendance Distribution")
    dist = predictions["prediction"].value_counts().reindex(["Attend", "No Show"]).fillna(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.0), facecolor="#0f1b2d")
    ax.set_facecolor("#0f1b2d")
    ax.pie(
        dist.values,
        labels=dist.index,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=["#22c55e", "#f87171"],
        wedgeprops=dict(width=.42, edgecolor="#0f1b2d"),
        textprops={"color": "#f8fafc", "fontsize": 10},
    )
    ax.text(0, 0, f"{total:,}\nTotal", ha="center", va="center", color="#e5e7eb", fontsize=10, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ch2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk Overview")
    risk_order = ["Low No-Show Risk", "Medium Risk", "High No-Show Risk"]
    risk = predictions["risk_level"].value_counts().reindex(risk_order).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.0), facecolor="#0f1b2d")
    ax2.set_facecolor("#0f1b2d")
    ax2.barh(risk.index, risk.values, color=["#22c55e", "#facc15", "#f87171"])
    ax2.tick_params(axis="x", colors="#cbd5e1", labelsize=9)
    ax2.tick_params(axis="y", colors="#cbd5e1", labelsize=9)
    ax2.grid(axis="x", alpha=.16)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>6) Feature Signal</div>", unsafe_allow_html=True)
try:
    st.dataframe(feature_signal_report(df), use_container_width=True, hide_index=True)
except Exception as exc:
    st.info(f"Feature signal unavailable: {exc}")

st.markdown("<div class='sec'>7) Prediction Output</div>", unsafe_allow_html=True)
show_cols = [
    c for c in [
        "Registration_ID",
        "Event_Type",
        "Ticket_Type",
        "Distance_KM",
        target,
        "attendance_probability",
        "prediction",
        "risk_level",
    ] if c in predictions.columns
]

if not show_cols:
    show_cols = predictions.columns[:8].tolist()

st.dataframe(
    predictions[show_cols].head(100),
    use_container_width=True,
    hide_index=True,
    column_config={
        "attendance_probability": st.column_config.ProgressColumn(
            "Attendance Probability",
            min_value=0,
            max_value=100,
            format="%.2f%%",
        )
    },
)

st.download_button(
    "Download prediction CSV",
    predictions.to_csv(index=False).encode("utf-8"),
    "smartevent_predictions.csv",
    "text/csv",
)

st.markdown(
    """
    <div class='info-box'>
    <b>Final note:</b><br>
    This upload-only version has no built-in dataset dependency. The dashboard opens first, then the uploaded file controls training,
    metrics, predictions, charts, and CSV download.
    </div>
    """,
    unsafe_allow_html=True,
)
