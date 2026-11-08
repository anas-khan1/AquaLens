@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting AquaLens Underwater AI Web Application...
echo ========================================================
python -m streamlit run app/streamlit_app.py
pause

