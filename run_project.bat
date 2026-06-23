@echo off
cd /d "%~dp0"

echo ===========================================
echo SmartEvent AI - Upload Only Version
echo ===========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH.
    pause
    exit /b 1
)

echo Installing requirements...
python -m pip install -r requirements.txt

echo.
echo Opening dashboard...
streamlit run app.py

pause
