import pandas as pd
from src.core import train_models, predict_dataframe

def main():
    df = pd.DataFrame({
        "Age": [22, 25, 41, 35, 29, 54, 46, 31, 23, 44, 37, 52, 28, 33, 39, 48, 21, 26, 42, 57],
        "Distance_KM": [2, 4, 48, 30, 8, 70, 42, 5, 3, 60, 35, 80, 9, 11, 45, 58, 2, 7, 40, 75],
        "Email_Opened": ["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No"],
        "Reminder_Clicked": ["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No"],
        "SMS_Confirmed": ["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No"],
        "Attendance_Rate": [0.9,0.85,0.1,0.2,0.8,0.0,0.15,0.95,0.9,0.1,0.7,0.05,0.8,0.75,0.2,0.1,0.95,0.9,0.25,0.05],
        "Payment_Status": ["Paid","Paid","Pending","Free","Paid","Pending","Free","Paid","Paid","Pending","Paid","Free","Paid","Paid","Pending","Free","Paid","Paid","Free","Pending"],
        "Attended": ["Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No"],
    })

    artifact = train_models(df)
    pred = predict_dataframe(df, artifact)
    assert "attendance_probability" in pred.columns
    assert "prediction" in pred.columns
    assert "risk_level" in pred.columns
    print("UPLOAD-ONLY PROJECT TEST PASSED")
    print(artifact["metrics"].to_string(index=False))

if __name__ == "__main__":
    main()
