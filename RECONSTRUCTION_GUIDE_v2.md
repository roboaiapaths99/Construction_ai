# Backend & Frontend Reconstruction v2.0

## Overview
Completely rebuilt backend and frontend with:
- ✅ Simplified architecture
- ✅ Error handling
- ✅ YOLO model integration
- ✅ Clean React UI
- ✅ No complex workers or conflicts

## Directory Structure

```
ai_construction_system/
├── backend_simple/
│   ├── main.py              # FastAPI main server
│   ├── requirements.txt      # Python dependencies
│   └── violations/           # Violation storage
├── frontend_simple/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   └── components/
│   │       ├── CameraFeed.js
│   │       ├── DetectionResults.js
│   │       └── StatusPanel.js
│   ├── package.json
└── ai/
    └── models/
        └── yolov8n.pt       # YOLO model (kept from original)
```

## Backend Setup

### Step 1: Install Python Dependencies
```bash
cd backend_simple
pip install -r requirements.txt
```

### Step 2: Run the Backend Server
```bash
python main.py
```

Expected output:
```
🚀 Starting application...
Loading model from: [path]/ai/models/yolov8n.pt
✅ Model loaded successfully
Starting server on 0.0.0.0:8000
```

### Backend Endpoints

- `GET /health` - Health check
- `GET /status` - Get system status
- `GET /camera/start` - Start webcam
- `GET /camera/stop` - Stop webcam
- `GET /camera/frame` - Get current frame
- `POST /detect` - Run detection on uploaded frame

## Frontend Setup

### Step 1: Install Dependencies
```bash
cd frontend_simple
npm install
```

### Step 2: Run Development Server
```bash
npm start
```

The app will open at: `http://localhost:3000`

### Frontend Features

- 📹 Live Camera Feed
- 📊 Detection Results
- ⚙️ System Status
- 🎮 Camera Controls

## Quick Start

### Terminal 1 - Backend
```bash
cd backend_simple
pip install -r requirements.txt
python main.py
```

### Terminal 2 - Frontend
```bash
cd frontend_simple
npm install
npm start
```

### Terminal 3 - Test Backend (Optional)
```bash
# Check health
curl http://localhost:8000/health

# Check status
curl http://localhost:8000/status
```

## Troubleshooting

### Issue: Model not loading
- Verify path: `ai/models/yolov8n.pt`
- Check YOLO installation: `pip show ultralytics`

### Issue: Camera not starting
- Ensure webcam is connected
- Check if another app is using the camera
- Try disconnecting and reconnecting USB camera

### Issue: Frontend can't reach backend
- Verify backend is running on `http://localhost:8000`
- Check CORS settings in `backend_simple/main.py`

### Issue: Port already in use
- Backend: Change `APP_PORT = 8000` in main.py
- Frontend: Use `PORT=3001 npm start`

## API Examples

### Start Camera
```bash
curl http://localhost:8000/camera/start
```

### Get Frame
```bash
curl http://localhost:8000/camera/frame -o frame.json
```

### Run Detection
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/detect
```

## What's Different v1 → v2

| Feature | v1 | v2 |
|---------|----|----|
| Architecture | Complex | Simple |
| Workers | Multiple | None |
| Database | MongoDB | File-based |
| Error Handling | Limited | Comprehensive |
| CORS | Restricted | Open |
| Model Loading | Conditional | Automatic |
| Components | Many | Essential |
| Setup Complexity | High | Low |

## Next Steps

Once you confirm the backend and frontend are working:
1. Add advanced detection features
2. Implement database storage
3. Add authentication
4. Deploy to production

## Support

For issues, check:
1. Terminal output for error messages
2. Browser console for frontend errors
3. Backend status endpoint
