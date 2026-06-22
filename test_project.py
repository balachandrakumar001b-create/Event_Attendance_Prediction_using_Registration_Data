from src.core import (
    load_default_dataset,
    train_models,
    save_artifact,
    load_artifact,
    predict_dataframe,
    feature_signal_report,
)

def main():
    df = load_default_dataset()
    assert len(df) >= 20, "Dataset too small."
    assert "Attended" in df.columns, "Target column missing."

    artifact = train_models(df)
    path = save_artifact(artifact)
    assert path.exists(), "Model was not saved."

    loaded = load_artifact(path)
    pred = predict_dataframe(df.head(200), loaded)
    assert {"attendance_probability", "prediction", "risk_level"}.issubset(pred.columns)
    assert pred["attendance_probability"].between(0, 100).all()

    signal = feature_signal_report(df)
    assert not signal.empty

    print("ALL TESTS PASSED")
    print(f"Rows: {len(df):,}")
    print(f"Best model: {artifact['best_model']}")
    print(artifact["metrics"].to_string(index=False))
    print(f"Model saved: {path}")

if __name__ == "__main__":
    main()
