@echo off
REM Production Startup Script
REM Starts all services in the correct order

echo ========================================
echo AI Construction Safety System
echo Production Architecture Startup
echo ========================================
echo.

REM 1. Start MediaMTX (Media Server)
echo [1/4] Starting MediaMTX...
start "MediaMTX" mediamtx.exe
timeout /t 3 /nobreak >nul
echo MediaMTX started on ports:
echo   - RTSP: 8554
echo   - WebRTC: 8889
echo   - HLS: 8888
echo.

REM 2. Start AI Worker (Face Recognition)
echo [2/4] Starting AI Worker...
cd backend
start "AI Worker" python ai_worker_standalone.py
cd ..
timeout /t 2 /nobreak >nul
echo AI Worker started on port 8001
echo.

REM 3. Start Backend API
echo [3/4] Starting Backend API...
cd backend
start "Backend API" python server_backup.py
cd ..
timeout /t 2 /nobreak >nul
echo Backend API started on port 8080
echo.

REM 4. Start Frontend
echo [4/4] Starting Frontend...
cd frontend
start "Frontend" npm start
cd ..
echo Frontend starting...
echo.

echo ========================================
echo All services started!
echo ========================================
echo.
echo Access URLs:
echo   - Frontend: http://localhost:4000
echo   - Backend API: http://localhost:8080
echo   - AI Worker: http://localhost:8001
echo   - MediaMTX WebRTC: http://localhost:8889/sitecam
echo   - MediaMTX HLS: http://localhost:8888/sitecam/index.m3u8
echo.
echo Press any key to stop all services...
pause >nul

REM Stop all services
echo.
echo Stopping services...
taskkill /F /IM mediamtx.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
echo All services stopped.
