#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python test_project.py
streamlit run app.py
