@echo off
REM Frontend Startup Script
echo 🎨 Starting AI Safety Monitoring Frontend...
echo.

cd /d "%~dp0"

echo Checking Node.js installation...
node --version

echo.
echo Installing dependencies (if needed)...
call npm install

echo.
echo Starting React dev server on http://localhost:3000
echo.

call npm start
