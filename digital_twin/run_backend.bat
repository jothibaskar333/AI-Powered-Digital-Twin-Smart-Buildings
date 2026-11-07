@echo off
REM Starts the FastAPI backend (simulation loop + API + 3D twin at http://localhost:8000/twin/)
call "%~dp0venv\Scripts\activate.bat"
cd /d "%~dp0backend"
uvicorn main:app --port 8000
