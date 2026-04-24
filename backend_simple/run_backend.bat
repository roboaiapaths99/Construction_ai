@echo off
REM Backend Startup Script
echo 🚀 Starting AI Safety Monitoring Backend...
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version

echo.
echo Starting FastAPI server on http://localhost:8000
echo.

python main.py

echo.
echo To stop the server, press Ctrl+C
pause
