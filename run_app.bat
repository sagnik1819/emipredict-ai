@echo off
title EMIPredict AI Platform
echo =======================================================
echo          Starting EMIPredict AI Platform
echo =======================================================
echo.
cd /d "%~dp0"
echo Launching Streamlit on http://localhost:8501 ...
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501
pause
