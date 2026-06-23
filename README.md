# SmartEvent AI - Upload Only Version

This version does not need any preinstalled dataset.

The application opens first. Then you upload CSV or Excel from the browser.

## Run

Double-click:

```text
run_project.bat
```

## Upload requirement

Your uploaded file must contain:

```text
Attended
```

or:

```text
attendance
```

Accepted target values:

```text
Yes / No
1 / 0
```

## Recommended columns for high accuracy

```text
Distance_KM
Reminder_Clicked
Email_Opened
SMS_Confirmed
Previous_No_Show_Count
Attendance_Rate
Ticket_Type
Weather
```

## Workflow

```text
Open app
↓
Upload CSV / Excel
↓
Train models
↓
Show Accuracy, F1 Score, ROC-AUC
↓
Predict attendance probability
↓
Show charts and risk level
↓
Download prediction CSV
```
