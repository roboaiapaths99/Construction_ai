# Production Architecture Documentation

## Overview

This system now follows a production-ready architecture with MediaMTX as the central media relay layer. This ensures stability, scalability, and proper separation of concerns.

## Architecture Diagram

```
┌─────────────┐
│ IP Camera   │ (RTSP/HTTP)
│  (H.264)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│      MediaMTX (Media Server)    │
│  - RTSP Input (TCP)             │
│  - WebRTC Output (Port 8889)    │
│  - HLS Output (Port 8888)        │
│  - RTSP Output (Port 8554)      │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
┌──────────────┐  ┌──────────────┐
│ AI Worker    │  │  Frontend    │
│ (Port 8001)  │  │  (Port 4000) │
│ - Face Rec   │  │ - WebRTC/HLS │
│ - Attendance │  │ - Dashboard  │
└──────┬───────┘  └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌──────────────┐
        │  Backend API │
        │  (Port 8080) │
        │ - Workers    │
        │ - Attendance │
        │ - Violations │
        └──────────────┘
```

## Key Components

### 1. MediaMTX (Media Server)
- **Purpose**: Central media relay and streaming server
- **Ports**:
  - RTSP: 8554
  - WebRTC: 8889
  - HLS: 8888
- **Configuration**: `mediamtx.yml`
- **Benefits**:
  - Single camera connection (no overload)
  - Browser-friendly outputs (WebRTC/HLS)
  - TCP transport for stability
  - Automatic reconnection

### 2. AI Worker (Standalone)
- **File**: `backend/ai_worker_standalone.py`
- **Port**: 8001
- **Purpose**: Face recognition and attendance marking
- **Features**:
  - Reads from MediaMTX RTSP stream
  - Processes every 3rd frame (configurable)
  - Auto-reconnect on failure
  - Health monitoring endpoints
  - Separate from backend API

### 3. Backend API
- **File**: `backend/server.py`
- **Port**: 8080
- **Purpose**: REST APIs for business logic
- **Responsibilities**:
  - Worker management
  - Attendance records
  - Violations/incidents
  - Face enrollment
  - Proxies AI worker stats

### 4. Frontend
- **Port**: 4000
- **Purpose**: User interface
- **Streaming**: Uses MediaMTX WebRTC (primary) with HLS fallback
- **Configuration**: `frontend/src/config/index.js`

## Configuration

### MediaMTX Configuration (`mediamtx.yml`)

```yaml
paths:
  sitecam:
    # IP Camera RTSP input - use TCP for stability
    source: rtsp://192.168.1.100:554/stream
    rtspTransport: tcp
    rtspAnyPort: yes
    sourceOnDemand: yes

  ai_worker:
    # AI worker reads from this path
    source: sitecam
    rtspTransport: tcp
```

### AI Worker Configuration

Environment variables:
- `MEDIAMTX_RTSP_URL`: RTSP URL to read from (default: `rtsp://localhost:8554/ai_worker`)
- `FRAME_SKIP`: Process every Nth frame (default: 3)
- `RECONNECT_DELAY`: Seconds to wait before reconnect (default: 5)
- `MAX_BAD_READS`: Max consecutive bad reads before reconnect (default: 10)

### Backend Configuration

Environment variables:
- `MEDIAMTX_RTSP_URL`: Camera RTSP URL (default: `rtsp://localhost:8554/sitecam`)
- `MEDIAMTX_WEBRTC_URL`: WebRTC URL (default: `http://localhost:8889/sitecam`)
- `MEDIAMTX_HLS_URL`: HLS URL (default: `http://localhost:8888/sitecam/index.m3u8`)
- `AI_WORKER_URL`: AI worker API URL (default: `http://localhost:8001`)

## Starting the System

### Production Startup

Run the startup script:
```bash
start_production.bat
```

This starts services in order:
1. MediaMTX (media server)
2. AI Worker (face recognition)
3. Backend API (REST APIs)
4. Frontend (user interface)

### Manual Startup

1. **Start MediaMTX**:
   ```bash
   mediamtx.exe
   ```

2. **Start AI Worker**:
   ```bash
   cd backend
   python ai_worker_standalone.py
   ```

3. **Start Backend**:
   ```bash
   cd backend
   python server.py
   ```

4. **Start Frontend**:
   ```bash
   cd frontend
   npm start
   ```

### Webcam Testing (No IP Camera)

Use the FFmpeg bridge to stream your webcam to MediaMTX:
```bash
webcam_to_mediamtx.bat
```

This streams webcam index 0 to MediaMTX path `webcam`.

## Camera Setup

### IP Camera Configuration

For production use, configure your IP camera with:
- **Codec**: H.264
- **Resolution**: 720p (1280x720) or 1080p (1920x1080)
- **FPS**: 10-15 fps (for AI processing)
- **Bitrate**: Moderate (2-4 Mbps for 720p)
- **Transport**: TCP (for stability)

Update `mediamtx.yml` with your camera's RTSP URL:
```yaml
paths:
  sitecam:
    source: rtsp://username:password@192.168.1.100:554/stream
    rtspTransport: tcp
```

## Health Monitoring

### AI Worker Endpoints

- `GET /health` - Basic health check
- `GET /stats` - Detailed statistics
- `POST /restart` - Restart the worker

### Backend Endpoints

- `GET /` - System status
- `GET /api/recognition/stats` - AI worker stats (proxied)
- `POST /api/recognition/restart` - Restart AI worker (proxied)

## Performance Optimization

### Frame Skipping
The AI worker processes every Nth frame (default: 3) to reduce CPU load while maintaining real-time performance.

### TCP Transport
RTSP over TCP is used by default for stability. UDP can be faster but is more sensitive to packet loss.

### Substream
For best performance, configure your camera to use a substream (lower resolution) for AI processing:
- Resolution: 640x360 or 1280x720
- FPS: 10-15
- Bitrate: 1-2 Mbps

## Troubleshooting

### Camera Not Connecting
1. Check RTSP URL in `mediamtx.yml`
2. Verify camera network connectivity
3. Check camera credentials
4. Ensure TCP transport is enabled

### AI Worker Not Processing
1. Check AI worker logs
2. Verify MediaMTX is running
3. Check `MEDIAMTX_RTSP_URL` environment variable
4. Ensure worker embeddings are loaded

### Frontend Not Streaming
1. Check MediaMTX is running
2. Verify WebRTC/HLS URLs in frontend config
3. Check browser console for errors
4. Try HLS fallback if WebRTC fails

## Benefits of This Architecture

1. **Single Camera Connection**: MediaMTX handles one connection to the camera, preventing overload
2. **Browser-Friendly**: WebRTC/HLS outputs work in all modern browsers
3. **Separation of Concerns**: Each service has a single responsibility
4. **Scalability**: Services can be scaled independently
5. **Stability**: TCP transport and auto-reconnect ensure reliable streaming
6. **Performance**: Frame skipping and substream reduce processing load
7. **Monitoring**: Health endpoints for all services

## Migration from Old Architecture

The old architecture had the backend directly accessing the camera and streaming MJPEG. The new architecture:

- **Removed**: Direct camera access in backend
- **Removed**: MJPEG streaming from backend
- **Added**: MediaMTX as media relay
- **Added**: Standalone AI worker
- **Changed**: Frontend to use WebRTC/HLS instead of MJPEG

To migrate:
1. Update `mediamtx.yml` with your camera URL
2. Start using `ai_worker_standalone.py` instead of integrated worker
3. Update frontend to use new streaming URLs
4. Remove old camera/stream code from backend
