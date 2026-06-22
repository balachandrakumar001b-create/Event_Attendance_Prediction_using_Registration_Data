@echo off
cd /d "%~dp0"

echo ===========================================
echo SmartEvent AI - Final Perfect UI
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
echo Running self-test...
python test_project.py
if errorlevel 1 (
    echo.
    echo Self-test failed. Fix the error shown above.
    pause
    exit /b 1
)

echo.
echo Starting dashboard...
streamlit run app.py

pause
