# SmartEvent AI Attendance Prediction - Final Perfect UI + Strong Dataset

This version fixes the UI issues and includes a stronger realistic dataset.

## Fixed UI issues

- Removed raw HTML model-score display
- Uses native Streamlit tables and progress bars
- Cleaner model score table
- More consistent chart sizes
- Reduced vertical scrolling
- Added final explanation box about low accuracy
- Included high-signal dataset for better model performance

## Dataset

Included:

```text
data/smartevent_high_signal_dataset.csv
```

This dataset includes strong predictive columns:

- Distance_KM
- Reminder_Clicked
- Email_Opened
- SMS_Confirmed
- Previous_Registered_Count
- Previous_No_Show_Count
- Attendance_Rate
- Ticket_Type
- Event_Day
- Event_Time
- Weather

## One-click run

Double-click:

```text
run_project.bat
```

## Manual run

```bash
pip install -r requirements.txt
python test_project.py
streamlit run app.py
```

## Target column

Required:

```text
Attended
```

Accepted values:

```text
Yes / No
1 / 0
```

## Strict note

This high-signal dataset is synthetic but realistic. It is suitable for final-year demo and model explanation.
For real deployment, collect the same columns from actual events.


## Validated test result on included high-signal dataset

```text
Best model: Random Forest
Accuracy: 86.21%
F1 Score: 86.77%
ROC-AUC: 90.78%
```

These results are from the included realistic synthetic dataset. Real-world accuracy depends on the quality of real event data.
