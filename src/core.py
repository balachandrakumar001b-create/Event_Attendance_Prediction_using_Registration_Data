from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


TARGET_CANDIDATES = [
    "attended",
    "attendance",
    "actual_attended",
    "actual_attendance",
    "target",
    "label",
]


def clean_col(col: str) -> str:
    return str(col).strip().replace(" ", "_").replace("-", "_").replace("/", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_col(c) for c in out.columns]
    return out


def read_table(file_obj: Any) -> pd.DataFrame:
    name = str(getattr(file_obj, "name", file_obj)).lower()

    if name.endswith(".csv"):
        return normalize_columns(pd.read_csv(file_obj))

    if name.endswith((".xlsx", ".xls")):
        return normalize_columns(pd.read_excel(file_obj))

    raise ValueError("Unsupported file type. Upload CSV or Excel only.")


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for target in TARGET_CANDIDATES:
        if target in lower:
            return lower[target]
    return None


def map_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)

    s = y.astype(str).str.strip().str.lower()

    yes_values = {
        "yes", "y", "1", "true", "attend", "attended",
        "present", "came", "joined", "participated"
    }

    no_values = {
        "no", "n", "0", "false", "absent", "no show",
        "no_show", "not attended", "not_attended", "missed"
    }

    mapped = s.map(lambda v: 1 if v in yes_values else 0 if v in no_values else np.nan)

    if mapped.isna().any():
        bad = sorted(s[mapped.isna()].dropna().unique().tolist())[:10]
        raise ValueError(
            "Target column has unsupported values. Use Yes/No or 1/0. "
            f"Unsupported examples: {bad}"
        )

    return mapped.astype(int)


def first_available_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def yes_no_to_num(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0})
        .fillna(0)
        .astype(int)
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)

    age = first_available_column(out, ["Age", "attendee_age"])
    distance = first_available_column(out, ["Distance_KM", "distance", "distance_km"])
    reg_days = first_available_column(out, ["Registration_Time_Days_Before", "registration_days_before", "days_before_event"])
    prev_registered = first_available_column(out, ["Previous_Registered_Count", "previous_registered"])
    prev_attended = first_available_column(out, ["Previous_Attendance_Count", "previous_attendance"])
    prev_no_show = first_available_column(out, ["Previous_No_Show_Count", "previous_no_show"])
    attendance_rate = first_available_column(out, ["Attendance_Rate", "attendance_rate"])
    email = first_available_column(out, ["Email_Opened", "email_opened"])
    reminder = first_available_column(out, ["Reminder_Clicked", "reminder_clicked"])
    sms = first_available_column(out, ["SMS_Confirmed", "sms_confirmed"])
    payment = first_available_column(out, ["Payment_Status", "payment_status", "payment"])
    ticket = first_available_column(out, ["Ticket_Type", "ticket_type", "ticket"])

    if age:
        age_num = pd.to_numeric(out[age], errors="coerce")
        out["age_group"] = pd.cut(
            age_num,
            bins=[0, 20, 30, 40, 50, 65, 120],
            labels=["<20", "21-30", "31-40", "41-50", "51-65", "65+"],
            include_lowest=True,
        ).astype(str)

    if distance:
        dist = pd.to_numeric(out[distance], errors="coerce").fillna(0)
        out["distance_bucket"] = pd.cut(
            dist,
            bins=[-1, 5, 15, 40, 10**9],
            labels=["very_near", "near", "medium", "far"],
            include_lowest=True,
        ).astype(str)
        out["distance_risk_score"] = np.clip(dist / 100.0, 0, 1)

    if reg_days:
        days = pd.to_numeric(out[reg_days], errors="coerce").fillna(0)
        out["registration_bucket"] = pd.cut(
            days,
            bins=[-1, 2, 7, 14, 30, 10**9],
            labels=["last_minute", "one_week", "two_weeks", "one_month", "early"],
            include_lowest=True,
        ).astype(str)
        out["early_registration_score"] = np.clip(days / 60.0, 0, 1)

    if prev_registered and prev_attended:
        reg = pd.to_numeric(out[prev_registered], errors="coerce").fillna(0)
        att = pd.to_numeric(out[prev_attended], errors="coerce").fillna(0)
        out["computed_attendance_rate"] = np.where(reg > 0, att / np.maximum(reg, 1), 0.5)
    elif attendance_rate:
        out["computed_attendance_rate"] = pd.to_numeric(out[attendance_rate], errors="coerce").fillna(0.5)
    else:
        out["computed_attendance_rate"] = 0.5

    if prev_registered and prev_no_show:
        reg = pd.to_numeric(out[prev_registered], errors="coerce").fillna(0)
        no_show = pd.to_numeric(out[prev_no_show], errors="coerce").fillna(0)
        out["computed_no_show_rate"] = np.where(reg > 0, no_show / np.maximum(reg, 1), 0)
    else:
        out["computed_no_show_rate"] = 0

    out["email_score"] = yes_no_to_num(out[email]) if email else 0
    out["reminder_score"] = yes_no_to_num(out[reminder]) if reminder else 0
    out["sms_score"] = yes_no_to_num(out[sms]) if sms else 0

    if payment:
        out["payment_score"] = (
            out[payment]
            .astype(str)
            .str.lower()
            .str.contains("paid|confirmed|success|premium|vip|yes|1", regex=True)
            .astype(int)
        )
    else:
        out["payment_score"] = 0

    out["engagement_score"] = (
        0.25 * pd.to_numeric(out["email_score"], errors="coerce").fillna(0)
        + 0.35 * pd.to_numeric(out["reminder_score"], errors="coerce").fillna(0)
        + 0.40 * pd.to_numeric(out["sms_score"], errors="coerce").fillna(0)
    )

    out["behavior_score"] = (
        0.55 * pd.to_numeric(out["computed_attendance_rate"], errors="coerce").fillna(0.5)
        + 0.25 * pd.to_numeric(out["engagement_score"], errors="coerce").fillna(0)
        + 0.20 * pd.to_numeric(out["payment_score"], errors="coerce").fillna(0)
    )

    if ticket and payment:
        out["ticket_payment_group"] = out[ticket].astype(str) + "_" + out[payment].astype(str)

    return out


def split_xy(df: pd.DataFrame, require_target: bool = True):
    df2 = add_engineered_features(df)
    target_col = find_target_column(df2)

    if require_target and target_col is None:
        raise ValueError("Training requires a target column: Attended or attendance with Yes/No or 1/0.")

    y = None
    if target_col is not None:
        y = map_target(df2[target_col])
        df2 = df2.drop(columns=[target_col])

    drop_cols = []
    for col in df2.columns:
        lc = col.lower()
        if lc in {"registration_id", "attendee_id", "user_id", "participant_id", "name", "email", "phone", "mobile"}:
            drop_cols.append(col)
        elif df2[col].dtype == object and df2[col].nunique(dropna=True) > min(1000, max(20, len(df2) * 0.30)):
            drop_cols.append(col)

    X = df2.drop(columns=drop_cols, errors="ignore").dropna(axis=1, how="all")

    if X.empty:
        raise ValueError("No usable feature columns found after removing ID/name columns.")

    return X, y, target_col


def make_preprocessor(X: pd.DataFrame):
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def train_models(df: pd.DataFrame, max_rows: int = 15000):
    X_all, y_all, target_col = split_xy(df, require_target=True)

    if len(X_all) < 20:
        raise ValueError("Dataset too small. Upload at least 20 rows.")
    if y_all.nunique() < 2:
        raise ValueError("Target column must contain both classes: Yes and No.")

    # For large files, train on a stratified sample so the app opens fast.
    if len(X_all) > max_rows:
        work = X_all.copy()
        work["__target__"] = y_all.values
        parts = []
        for _, group in work.groupby("__target__"):
            sample_size = max(1, int(max_rows * len(group) / len(work)))
            parts.append(group.sample(min(len(group), sample_size), random_state=42))
        work = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        y = work.pop("__target__").astype(int)
        X = work
    else:
        X = X_all
        y = y_all

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(X_train)),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, pred) * 100, 2),
                "F1 Score": round(f1_score(y_test, pred, average="weighted") * 100, 2),
                "ROC-AUC": round(roc_auc_score(y_test, prob) * 100, 2),
            }
        )
        fitted[name] = pipe

    metrics = pd.DataFrame(results)
    score = 0.45 * metrics["Accuracy"] + 0.45 * metrics["F1 Score"] + 0.10 * metrics["ROC-AUC"]
    best_index = score.idxmax()
    best_name = str(metrics.loc[best_index, "Model"])
    best_pipeline = fitted[best_name]

    best_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])

    artifact = {
        "pipeline": best_pipeline,
        "best_model": best_name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_columns": X_all.columns.tolist(),
        "target_column": target_col,
        "rows_trained_from": int(len(X)),
        "total_rows_uploaded": int(len(df)),
    }

    return artifact


def align_for_prediction(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X, _, _ = split_xy(df, require_target=False)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    return X[feature_columns]


def risk_label(probability: float) -> str:
    if probability >= 0.70:
        return "Low No-Show Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "High No-Show Risk"


def predict_dataframe(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    X = align_for_prediction(df, artifact["feature_columns"])
    probability = artifact["pipeline"].predict_proba(X)[:, 1]

    output = normalize_columns(df).copy()
    output["attendance_probability"] = np.round(probability * 100, 2)
    output["prediction"] = np.where(probability >= 0.50, "Attend", "No Show")
    output["risk_level"] = [risk_label(float(p)) for p in probability]

    return output


def feature_signal_report(df: pd.DataFrame) -> pd.DataFrame:
    X, y, _ = split_xy(df, require_target=True)

    rows = []
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            corr = pd.to_numeric(X[column], errors="coerce").corr(y)
            signal = 0 if pd.isna(corr) else abs(float(corr))
            signal_type = "correlation"
        else:
            table = pd.crosstab(X[column].astype(str), y, normalize="index")
            signal = float(table[1].max() - table[1].min()) if 1 in table.columns else 0
            signal_type = "group spread"

        rows.append(
            {
                "Feature": column,
                "Signal Type": signal_type,
                "Signal": round(signal, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("Signal", ascending=False).head(12)
