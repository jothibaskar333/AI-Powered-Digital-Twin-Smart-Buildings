@echo off
REM Starts the Streamlit dashboard at http://localhost:8501 (start the backend first)
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0dashboard"
streamlit run app.py
