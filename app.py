from __future__ import annotations

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.core import (
    load_default_dataset,
    load_artifact,
    save_artifact,
    train_models,
    predict_dataframe,
    read_table,
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
.card {
    border: 1px solid rgba(148,163,184,.22);
    background: rgba(15,27,45,.94);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 14px 38px rgba(0,0,0,.16);
}
.sec {
    font-size: 1.35rem;
    font-weight: 850;
    margin-top: 1.35rem;
    margin-bottom: .75rem;
}
.status {
    display:inline-flex;
    align-items:center;
    gap:8px;
    border:1px solid rgba(148,163,184,.25);
    background:rgba(15,27,45,.82);
    padding:8px 14px;
    border-radius:999px;
    font-size:.9rem;
}
.dot {
    width:8px;
    height:8px;
    border-radius:99px;
    background:#22c55e;
    box-shadow:0 0 12px #22c55e;
    display:inline-block;
}
.warn {
    background:rgba(248,113,113,.13);
    border:1px solid rgba(248,113,113,.35);
    border-left:4px solid #f87171;
    border-radius:14px;
    padding:13px 16px;
    margin:8px 0 14px;
    color:#fecaca;
}
.ok {
    background:rgba(34,197,94,.13);
    border:1px solid rgba(34,197,94,.30);
    border-left:4px solid #22c55e;
    border-radius:14px;
    padding:13px 16px;
    margin:8px 0 14px;
    color:#bbf7d0;
}
.info-box {
    background:rgba(96,165,250,.12);
    border:1px solid rgba(96,165,250,.32);
    border-left:4px solid #60a5fa;
    border-radius:14px;
    padding:13px 16px;
    margin:16px 0 10px;
}
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


def pct(v):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{float(v):.2f}%"


def metric(artifact, name):
    m = artifact["metrics"]
    r = m[m["Model"] == artifact["best_model"]]
    return None if r.empty else float(r.iloc[0][name])


@st.cache_resource(show_spinner=False)
def get_artifact():
    return load_artifact()


def retrain_default():
    art = train_models(load_default_dataset())
    save_artifact(art)
    return art


with st.sidebar:
    st.markdown("## 📅 SmartEvent AI")
    st.markdown("<span class='small'>Clean attendance prediction dashboard</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Controls")
    st.markdown("<span class='small'>Target: <b>Attended</b> or <b>attendance</b><br>Values: Yes/No or 1/0.</span>", unsafe_allow_html=True)
    retrain = st.button("🔁 Retrain included dataset", use_container_width=True)
    st.markdown("---")
    st.markdown("### Dashboard")
    st.markdown(
        "1. Overview  \n"
        "2. Model score  \n"
        "3. Upload data  \n"
        "4. Prediction summary  \n"
        "5. Simple charts  \n"
        "6. Confusion matrix  \n"
        "7. Prediction output"
    )
    st.markdown("---")
    st.caption("This version includes a stronger realistic dataset for better model learning.")

if retrain:
    with st.spinner("Retraining model..."):
        artifact = retrain_default()
    st.success("Retrained successfully.")
else:
    artifact = get_artifact()

best = artifact["best_model"]
acc = metric(artifact, "Accuracy")
f1 = metric(artifact, "F1 Score")
roc = metric(artifact, "ROC-AUC")

head_l, head_r = st.columns([0.82, 0.18])
with head_l:
    st.title("SmartEvent AI Attendance Prediction")
    st.markdown("<span class='small'>Upload registration data and view simple attendance insights.</span>", unsafe_allow_html=True)
with head_r:
    st.markdown("<br><div class='status'><span class='dot'></span> Model Ready</div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>1) Overview</div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>Best model</div>"
        f"<div class='kpi-value'>{best}</div><div class='kpi-sub'>{len(artifact['feature_columns'])} features used</div></div>",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>Accuracy</div>"
        f"<div class='kpi-value' style='color:#22c55e'>{pct(acc)}</div><div class='kpi-sub'>Correct predictions</div></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>F1 Score</div>"
        f"<div class='kpi-value' style='color:#a78bfa'>{pct(f1)}</div><div class='kpi-sub'>Balanced performance</div></div>",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"<div class='kpi'><div class='kpi-label'>ROC-AUC</div>"
        f"<div class='kpi-value' style='color:#60a5fa'>{pct(roc)}</div><div class='kpi-sub'>Ranking quality</div></div>",
        unsafe_allow_html=True,
    )

if roc is not None and roc < 55:
    st.markdown(
        "<div class='warn'><b>Data warning:</b> ROC-AUC is below 55%. The app works, but this dataset has weak predictive signal.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='ok'><b>Signal check:</b> Dataset has usable predictive signal. The model can learn meaningful attendance patterns.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='sec'>2) Model Score</div>", unsafe_allow_html=True)
m1, m2 = st.columns([0.68, 0.32])
with m1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model performance comparison")
    metrics_df = artifact["metrics"].copy()
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.2f%%"),
            "F1 Score": st.column_config.ProgressColumn("F1 Score", min_value=0, max_value=100, format="%.2f%%"),
            "ROC-AUC": st.column_config.ProgressColumn("ROC-AUC", min_value=0, max_value=100, format="%.2f%%"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

with m2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Feature signal check")
    try:
        signal = feature_signal_report(load_default_dataset()).head(7)
        st.dataframe(signal, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(str(e))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>3) Upload CSV / Excel for Prediction</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload registration file", type=["csv", "xlsx", "xls"])
df = read_table(uploaded) if uploaded else load_default_dataset()
target = find_target_column(df)

if target is not None:
    with st.spinner("Training and predicting on uploaded dataset..."):
        active_artifact = train_models(df)
else:
    active_artifact = artifact

pred_df = predict_dataframe(df, active_artifact)
total = len(pred_df)
attend_count = int((pred_df["prediction"] == "Attend").sum())
noshow_count = total - attend_count
avg_prob = float(pred_df["attendance_probability"].mean())

st.markdown(f"<span class='small'>Rows loaded: <b>{total:,}</b> | Columns: <b>{len(df.columns):,}</b></span>", unsafe_allow_html=True)

st.markdown("<div class='sec'>4) Prediction Summary</div>", unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Total registrations</div><div class='kpi-value'>{total:,}</div><div class='kpi-sub'>Uploaded records</div></div>", unsafe_allow_html=True)
with s2:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Predicted attend</div><div class='kpi-value' style='color:#22c55e'>{attend_count:,}</div><div class='kpi-sub'>{attend_count/total*100:.2f}%</div></div>", unsafe_allow_html=True)
with s3:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Predicted no-show</div><div class='kpi-value' style='color:#f87171'>{noshow_count:,}</div><div class='kpi-sub'>{noshow_count/total*100:.2f}%</div></div>", unsafe_allow_html=True)
with s4:
    st.markdown(f"<div class='kpi'><div class='kpi-label'>Average probability</div><div class='kpi-value' style='color:#60a5fa'>{avg_prob:.2f}%</div><div class='kpi-sub'>Mean attend chance</div></div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>5) Simple Charts</div>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Predicted attendance distribution")
    dist = pred_df["prediction"].value_counts().reindex(["Attend", "No Show"]).fillna(0)
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
    ax.set_title("Attend vs No Show", color="#f8fafc", fontsize=12)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with ch2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Risk overview")
    risk_order = ["Low No-Show Risk", "Medium Risk", "High No-Show Risk"]
    risk = pred_df["risk_level"].value_counts().reindex(risk_order).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.0), facecolor="#0f1b2d")
    ax2.set_facecolor("#0f1b2d")
    ax2.barh(risk.index, risk.values, color=["#22c55e", "#facc15", "#f87171"])
    ax2.tick_params(axis="x", colors="#cbd5e1", labelsize=9)
    ax2.tick_params(axis="y", colors="#cbd5e1", labelsize=9)
    ax2.grid(axis="x", alpha=.16)
    ax2.set_xlabel("Count", color="#cbd5e1")
    ax2.set_title("No-show risk levels", color="#f8fafc", fontsize=12)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='sec'>6) Confusion Matrix</div>", unsafe_allow_html=True)
if target and target in pred_df.columns:
    actual = pred_df[target].astype(str).str.lower().str.strip().map({
        "yes": "Attend", "y": "Attend", "1": "Attend", "true": "Attend", "attended": "Attend",
        "no": "No Show", "n": "No Show", "0": "No Show", "false": "No Show", "absent": "No Show",
    }).fillna(pred_df[target].astype(str))
    cm = pd.crosstab(actual, pred_df["prediction"], rownames=["Actual"], colnames=["Predicted"])
    st.dataframe(cm, use_container_width=True)
else:
    st.info("Confusion matrix needs an actual target column such as Attended or attendance.")

st.markdown("<div class='sec'>7) Prediction Output</div>", unsafe_allow_html=True)
show_cols = [
    c for c in [
        "Registration_ID", "Event_Type", "Ticket_Type", "Distance_KM", "Attended",
        "attendance_probability", "prediction", "risk_level"
    ] if c in pred_df.columns
]
if not show_cols:
    show_cols = pred_df.columns[:8].tolist()

st.dataframe(
    pred_df[show_cols].head(100),
    use_container_width=True,
    hide_index=True,
    column_config={
        "attendance_probability": st.column_config.ProgressColumn(
            "Attendance Probability", min_value=0, max_value=100, format="%.2f%%"
        )
    },
)

st.download_button(
    "Download prediction CSV",
    pred_df.to_csv(index=False).encode("utf-8"),
    "smartevent_predictions.csv",
    "text/csv",
)

st.markdown(
    """
    <div class='info-box'>
    <b>Why accuracy can be low:</b><br>
    Dataset quality controls model quality. Weak columns like only age, gender, and event type cannot strongly predict attendance.
    For higher accuracy, use stronger columns such as <b>Distance_KM</b>, <b>Reminder_Clicked</b>, <b>Email_Opened</b>,
    <b>SMS_Confirmed</b>, <b>Previous_No_Show_Count</b>, and <b>Attendance_Rate</b>.
    </div>
    """,
    unsafe_allow_html=True,
)
