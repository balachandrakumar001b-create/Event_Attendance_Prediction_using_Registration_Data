from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


TARGETS = ["attended", "attendance", "actual_attended", "target", "label"]


def root() -> Path:
    return Path(__file__).resolve().parent.parent


def clean_col(c: str) -> str:
    return str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_col(c) for c in out.columns]
    return out


def read_table(path_or_file: Any) -> pd.DataFrame:
    name = str(getattr(path_or_file, "name", path_or_file)).lower()
    if name.endswith(".csv"):
        return normalize_columns(pd.read_csv(path_or_file))
    if name.endswith((".xlsx", ".xls")):
        return normalize_columns(pd.read_excel(path_or_file))
    raise ValueError("Upload CSV or Excel only.")


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for t in TARGETS:
        if t in lower:
            return lower[t]
    return None


def map_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)

    s = y.astype(str).str.strip().str.lower()
    yes = {"yes", "y", "1", "true", "attend", "attended", "present", "came", "joined"}
    no = {"no", "n", "0", "false", "absent", "no show", "no_show", "not attended", "missed"}

    out = s.map(lambda v: 1 if v in yes else 0 if v in no else np.nan)
    if out.isna().any():
        bad = sorted(s[out.isna()].dropna().unique().tolist())[:8]
        raise ValueError(f"Unsupported target values: {bad}. Use Yes/No or 1/0.")
    return out.astype(int)


def first_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def yes_no_to_num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().map({
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0
    }).fillna(0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)

    age = first_col(out, ["Age", "attendee_age"])
    reg_days = first_col(out, ["Registration_Time_Days_Before", "registration_days_before", "days_before_event"])
    prev_reg = first_col(out, ["Previous_Registered_Count", "previous_registered"])
    prev_att = first_col(out, ["Previous_Attendance_Count", "previous_attendance"])
    prev_no = first_col(out, ["Previous_No_Show_Count", "previous_no_show"])
    att_rate = first_col(out, ["Attendance_Rate", "attendance_rate"])
    distance = first_col(out, ["Distance_KM", "distance", "distance_km"])
    email = first_col(out, ["Email_Opened", "email_opened"])
    reminder = first_col(out, ["Reminder_Clicked", "reminder_clicked"])
    sms = first_col(out, ["SMS_Confirmed", "sms_confirmed"])
    pay = first_col(out, ["Payment_Status", "payment", "paid_status"])
    event = first_col(out, ["Event_Type", "event_category", "category"])
    ticket = first_col(out, ["Ticket_Type", "ticket"])

    if age:
        a = pd.to_numeric(out[age], errors="coerce")
        out["age_group"] = pd.cut(
            a, [0, 20, 30, 40, 50, 65, 120],
            labels=["<20", "21-30", "31-40", "41-50", "51-65", "65+"],
            include_lowest=True
        ).astype(str)

    if reg_days:
        d = pd.to_numeric(out[reg_days], errors="coerce").fillna(0)
        out["registration_bucket"] = pd.cut(
            d, [-1, 2, 7, 14, 30, 10**9],
            labels=["last_minute", "one_week", "two_weeks", "one_month", "early"],
            include_lowest=True
        ).astype(str)
        out["early_registration_score"] = np.clip(d / 60.0, 0, 1)

    if distance:
        dist = pd.to_numeric(out[distance], errors="coerce").fillna(0)
        out["distance_bucket"] = pd.cut(
            dist, [-1, 5, 15, 40, 10**9],
            labels=["very_near", "near", "medium", "far"],
            include_lowest=True
        ).astype(str)
        out["distance_risk_score"] = np.clip(dist / 100.0, 0, 1)

    if prev_reg and prev_att:
        pr = pd.to_numeric(out[prev_reg], errors="coerce").fillna(0)
        pa = pd.to_numeric(out[prev_att], errors="coerce").fillna(0)
        out["computed_attendance_rate"] = np.where(pr > 0, pa / np.maximum(pr, 1), 0.5)
    elif att_rate:
        out["computed_attendance_rate"] = pd.to_numeric(out[att_rate], errors="coerce").fillna(0.5)
    else:
        out["computed_attendance_rate"] = 0.5

    if prev_no and prev_reg:
        pn = pd.to_numeric(out[prev_no], errors="coerce").fillna(0)
        pr = pd.to_numeric(out[prev_reg], errors="coerce").fillna(0)
        out["computed_no_show_rate"] = np.where(pr > 0, pn / np.maximum(pr, 1), 0)
    else:
        out["computed_no_show_rate"] = 0

    if email:
        out["email_score"] = yes_no_to_num(out[email])
    else:
        out["email_score"] = 0

    if reminder:
        out["reminder_score"] = yes_no_to_num(out[reminder])
    else:
        out["reminder_score"] = 0

    if sms:
        out["sms_score"] = yes_no_to_num(out[sms])
    else:
        out["sms_score"] = 0

    if pay:
        ps = out[pay].astype(str).str.lower()
        out["payment_score"] = ps.str.contains("paid|confirmed|success|premium|vip|yes|1", regex=True).astype(int)
    else:
        out["payment_score"] = 0

    out["engagement_score"] = (
        0.25 * out["email_score"] +
        0.35 * out["reminder_score"] +
        0.40 * out["sms_score"]
    )

    out["behavior_score"] = (
        0.55 * pd.to_numeric(out["computed_attendance_rate"], errors="coerce").fillna(0.5) +
        0.25 * out["engagement_score"] +
        0.20 * out["payment_score"]
    )

    if event and ticket:
        out["event_ticket_group"] = out[event].astype(str) + "_" + out[ticket].astype(str)

    return out


def split_xy(df: pd.DataFrame, require_target=True):
    df2 = add_features(df)
    target = find_target_column(df2)
    if require_target and target is None:
        raise ValueError("Training needs target column: Attended or attendance.")

    y = None
    if target:
        y = map_target(df2[target])
        df2 = df2.drop(columns=[target])

    drop = []
    for c in df2.columns:
        lc = c.lower()
        if lc in {"registration_id", "attendee_id", "user_id", "participant_id", "name", "email", "phone", "mobile"}:
            drop.append(c)
        elif df2[c].dtype == object and df2[c].nunique(dropna=True) > min(1000, max(20, len(df2) * 0.30)):
            drop.append(c)

    X = df2.drop(columns=drop, errors="ignore").dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable feature columns found.")
    return X, y, target


def make_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cat)
    ])
    return pre, num, cat


def safe_split(X, y):
    if len(X) < 20:
        raise ValueError("Need at least 20 rows.")
    if y.nunique() < 2:
        raise ValueError("Need both classes: Yes and No.")
    strat = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=strat)


def train_models(df: pd.DataFrame, max_rows: int = 12000):
    X_all, y_all, target = split_xy(df, True)

    if len(X_all) > max_rows:
        tmp = X_all.copy()
        tmp["__target__"] = y_all.values
        sampled = []
        for _, g in tmp.groupby("__target__"):
            n = max(1, int(max_rows * len(g) / len(tmp)))
            sampled.append(g.sample(min(len(g), n), random_state=42))
        tmp = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
        y = tmp.pop("__target__").astype(int)
        X = tmp
    else:
        X, y = X_all, y_all

    Xtr, Xt, ytr, yt = safe_split(X, y)
    pre, num, cat = make_preprocessor(Xtr)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xt)
        proba = pipe.predict_proba(Xt)[:, 1]
        auc = roc_auc_score(yt, proba) * 100
        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(yt, pred) * 100, 2),
            "F1 Score": round(f1_score(yt, pred, average="weighted") * 100, 2),
            "ROC-AUC": round(float(auc), 2),
        })
        fitted[name] = pipe

    metrics = pd.DataFrame(rows)
    best_idx = (0.45 * metrics["Accuracy"] + 0.45 * metrics["F1 Score"] + 0.10 * metrics["ROC-AUC"]).idxmax()
    best = str(metrics.loc[best_idx, "Model"])
    best_pipe = fitted[best]
    cm = confusion_matrix(yt, best_pipe.predict(Xt), labels=[0, 1])

    return {
        "pipeline": best_pipe,
        "best_model": best,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_columns": X_all.columns.tolist(),
        "target_column": target,
        "rows_trained_from": int(len(X)),
        "num_features": num,
        "cat_features": cat,
    }


def save_artifact(artifact, path: Optional[Path] = None) -> Path:
    path = path or root() / "model" / "attendance_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_default_dataset():
    return read_table(root() / "data" / "smartevent_high_signal_dataset.csv")


def load_artifact(path: Optional[Path] = None):
    path = path or root() / "model" / "attendance_model.pkl"
    if not path.exists():
        art = train_models(load_default_dataset())
        save_artifact(art, path)
        return art
    return joblib.load(path)


def align_for_prediction(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X, _, _ = split_xy(df, require_target=False)
    for c in features:
        if c not in X.columns:
            X[c] = np.nan
    return X[features]


def risk(prob: float) -> str:
    if prob >= 0.70:
        return "Low No-Show Risk"
    if prob >= 0.40:
        return "Medium Risk"
    return "High No-Show Risk"


def predict_dataframe(df: pd.DataFrame, artifact):
    X = align_for_prediction(df, artifact["feature_columns"])
    prob = artifact["pipeline"].predict_proba(X)[:, 1]
    out = normalize_columns(df).copy()
    out["attendance_probability"] = np.round(prob * 100, 2)
    out["prediction"] = np.where(prob >= 0.50, "Attend", "No Show")
    out["risk_level"] = [risk(float(p)) for p in prob]
    return out


def feature_signal_report(df: pd.DataFrame) -> pd.DataFrame:
    X, y, _ = split_xy(df, True)
    rows = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            corr = pd.to_numeric(X[c], errors="coerce").corr(y)
            signal = 0 if pd.isna(corr) else abs(float(corr))
            typ = "correlation"
        else:
            tab = pd.crosstab(X[c].astype(str), y, normalize="index")
            signal = float(tab[1].max() - tab[1].min()) if 1 in tab.columns else 0
            typ = "group spread"
        rows.append({"Feature": c, "Signal Type": typ, "Signal": round(signal, 4)})
    return pd.DataFrame(rows).sort_values("Signal", ascending=False).head(12)
