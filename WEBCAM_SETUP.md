# Webcam Setup Guide

This guide explains how to switch between the IP camera (192.168.1.36) and your laptop webcam for testing the attendance system.

## Quick Start - Use Laptop Webcam

### Step 1: Start the webcam stream
Open a **new terminal** and run:
```powershell
cd c:\Users\Lenovo\Desktop\ai_construction_system - Copy
python webcam_rtsp_stream.py
```

You should see:
```
============================================================
Laptop Webcam to RTSP Streamer
============================================================

Starting webcam capture and streaming to MediaMTX...
Stream URL: rtsp://127.0.0.1:8554/webcam

Press Ctrl+C to stop streaming
```

### Step 2: Update the backend to use webcam
In another terminal, set the environment variable and start the backend:
```powershell
cd c:\Users\Lenovo\Desktop\ai_construction_system - Copy\backend
$env:CAMERA_SOURCE = "webcam"
python server.py
```

### Step 3: Test the attendance system
- Open frontend: http://localhost:3000
- Face recognition should work with your webcam
- Detected faces should appear in the attendance log

---

## Switch Back to IP Camera

### Step 1: Stop the webcam stream (in the first terminal)
Press `Ctrl+C` in the terminal running `webcam_rtsp_stream.py`

### Step 2: Restart the backend with IP camera
```powershell
cd c:\Users\Lenovo\Desktop\ai_construction_system - Copy\backend
$env:CAMERA_SOURCE = "sitecam"
python server.py
```

---

## Troubleshooting

### Webcam not detected
- Check that your laptop webcam is working
- Try using it in Zoom or other video app first
- Ensure no other app is using the webcam exclusively

### No stream in MediaMTX
- Verify `docker-compose ps` shows both `mediamtx` and `ffmpeg-bridge` running
- Check `docker-compose logs mediamtx -n 20` for errors
- The ffmpeg-bridge (IP camera) should keep running; the webcam runs separately

### Backend won't connect to stream
- Make sure `webcam_rtsp_stream.py` is running when using webcam
- Check backend logs for connection errors
- Verify MediaMTX is accepting the stream: `docker-compose logs mediamtx -n 10`

---

## System Status Endpoints

After updating CAMERA_SOURCE and restarting the backend:

- **Camera Status**: `curl http://127.0.0.1:8002/api/cameras/status`
- **Active Stream URL**: Check the response - should show `/webcam` or `/sitecam`

---

## Next Steps

Once you're done testing with the webcam:
1. Stop the webcam stream
2. Set `CAMERA_SOURCE=sitecam` to resume the IP camera
3. Restart the backend
4. Continue with IP camera setup and configuration
