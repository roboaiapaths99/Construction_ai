# AI Construction Safety System - API Endpoints

## Base URL
```
Development: http://localhost:8002
Production: https://your-domain.com
```

## Authentication
Currently using basic authentication. In production, implement JWT tokens.

## Response Format
All responses follow this format:
```json
{
  "success": true,
  "data": {},
  "message": "Operation successful",
  "timestamp": "2026-03-27T17:20:00Z"
}
```

## Error Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {}
  },
  "timestamp": "2026-03-27T17:20:00Z"
}
```

---

## 🏥 Health & Status

### GET /health
Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "database": "connected",
  "ai_model": "loaded"
}
```

---

## 📊 Dashboard

### GET /dashboard/stats
Get dashboard statistics.

**Response:**
```json
{
  "total_workers": 45,
  "total_violations": 12,
  "active_alerts": 3,
  "connected_cameras": 4,
  "violations_today": 5,
  "alerts_today": 2
}
```

---

## 👷 Workers

### GET /workers
Get all workers with optional filtering.

**Query Parameters:**
- `status` (optional): Filter by status (Active, Inactive, At Risk)
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "workers": [
    {
      "id": 1,
      "name": "John Doe",
      "role": "Construction Worker",
      "status": "Active",
      "location": "Building Site A",
      "created_at": "2026-03-27T10:00:00Z",
      "updated_at": "2026-03-27T17:20:00Z"
    }
  ],
  "total": 45,
  "limit": 50,
  "offset": 0
}
```

### GET /workers/{id}
Get specific worker details.

**Response:**
```json
{
  "id": 1,
  "name": "John Doe",
  "role": "Construction Worker", 
  "status": "Active",
  "location": "Building Site A",
  "violations_count": 2,
  "last_seen": "2026-03-27T17:15:00Z",
  "created_at": "2026-03-27T10:00:00Z"
}
```

### POST /workers
Create new worker.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "role": "Safety Officer",
  "location": "Main Entrance"
}
```

---

## ⚠️ Violations

### GET /violations
Get all violations with filtering.

**Query Parameters:**
- `status` (optional): Filter by status (open, resolved, ignored)
- `type` (optional): Filter by violation type
- `camera` (optional): Filter by camera name
- `date_from` (optional): Start date (YYYY-MM-DD)
- `date_to` (optional): End date (YYYY-MM-DD)
- `limit` (optional): Number of results (default: 50)

**Response:**
```json
{
  "violations": [
    {
      "id": 1,
      "worker_id": 1,
      "worker_name": "John Doe",
      "camera_name": "Main Entrance",
      "violation_type": "No Helmet",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 300
      },
      "image_path": "/data/images/violations/violation_20260327_172000.jpg",
      "status": "open",
      "timestamp": "2026-03-27T17:20:00Z"
    }
  ],
  "total": 12,
  "limit": 50
}
```

### GET /violations/{id}
Get specific violation details.

**Response:**
```json
{
  "id": 1,
  "worker_id": 1,
  "worker_name": "John Doe",
  "camera_name": "Main Entrance",
  "violation_type": "No Helmet",
  "confidence": 0.95,
  "bbox": {
    "x": 100,
    "y": 100,
    "width": 200,
    "height": 300
  },
  "image_path": "/data/images/violations/violation_20260327_172000.jpg",
  "status": "open",
  "timestamp": "2026-03-27T17:20:00Z",
  "resolved_at": null
}
```

### PUT /violations/{id}/status
Update violation status.

**Request Body:**
```json
{
  "status": "resolved",
  "notes": "Worker provided with helmet"
}
```

---

## 🚨 Alerts

### GET /alerts
Get all alerts with filtering.

**Query Parameters:**
- `level` (optional): Filter by level (low, medium, high)
- `status` (optional): Filter by status (active, acknowledged, resolved)
- `limit` (optional): Number of results (default: 50)

**Response:**
```json
{
  "alerts": [
    {
      "id": 1,
      "message": "Worker without safety vest detected",
      "level": "medium",
      "violation_id": 1,
      "camera_name": "Main Entrance",
      "status": "active",
      "created_at": "2026-03-27T17:20:00Z",
      "resolved_at": null
    }
  ],
  "total": 3,
  "limit": 50
}
```

### GET /alerts/{id}
Get specific alert details.

### PUT /alerts/{id}/status
Update alert status.

**Request Body:**
```json
{
  "status": "acknowledged",
  "notes": "Safety team notified"
}
```

---

## 📹 Cameras

### GET /cameras
Get all cameras.

**Response:**
```json
{
  "cameras": [
    {
      "id": 1,
      "name": "Main Entrance",
      "ip": "192.168.1.100",
      "location": "Front Gate",
      "status": "active",
      "created_at": "2026-03-27T10:00:00Z"
    },
    {
      "id": 2,
      "name": "Construction Area",
      "ip": "192.168.1.101", 
      "location": "Building Site",
      "status": "active",
      "created_at": "2026-03-27T10:00:00Z"
    }
  ]
}
```

### POST /cameras
Add new camera.

**Request Body:**
```json
{
  "name": "Parking Lot",
  "ip": "192.168.1.102",
  "location": "Employee Parking"
}
```

### PUT /cameras/{id}/status
Update camera status.

**Request Body:**
```json
{
  "status": "maintenance"
}
```

---

## 🤖 AI Detection

### POST /detect_base64
Perform AI detection on base64 encoded image.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 300
      }
    },
    {
      "class_id": 1,
      "class_name": "hard_hat",
      "confidence": 0.87,
      "bbox": {
        "x": 150,
        "y": 50,
        "width": 80,
        "height": 60
      }
    }
  ],
  "violations": [
    {
      "type": "no_safety_vest",
      "severity": "medium",
      "confidence": 0.95,
      "person_id": 1,
      "bbox": {
        "x": 100,
        "y": 100,
        "width": 200,
        "height": 300
      }
    }
  ],
  "processing_time_ms": 45
}
```

### GET /ai/model/info
Get AI model information.

**Response:**
```json
{
  "model_name": "yolov8n.pt",
  "model_type": "YOLOv8",
  "input_size": "640x640",
  "classes": ["person", "hard_hat", "safety_vest"],
  "confidence_threshold": 0.4,
  "iou_threshold": 0.5,
  "device": "cpu",
  "loaded_at": "2026-03-27T17:20:00Z"
}
```

---

## 📈 Statistics & Analytics

### GET /stats/violations
Get violation statistics.

**Query Parameters:**
- `period` (optional): Time period (today, week, month, year)
- `type` (optional): Violation type filter

**Response:**
```json
{
  "period": "week",
  "total_violations": 25,
  "by_type": {
    "no_helmet": 15,
    "no_vest": 8,
    "multiple_violations": 2
  },
  "by_day": [
    {"date": "2026-03-21", "count": 3},
    {"date": "2026-03-22", "count": 5}
  ],
  "resolved_rate": 0.8
}
```

### GET /stats/workers
Get worker statistics.

**Response:**
```json
{
  "total_workers": 45,
  "active_workers": 40,
  "at_risk_workers": 5,
  "by_role": {
    "Construction Worker": 35,
    "Safety Officer": 5,
    "Site Manager": 5
  },
  "attendance_rate": 0.95
}
```

---

## 🔧 System Configuration

### GET /config
Get system configuration.

**Response:**
```json
{
  "ai_config": {
    "confidence_threshold": 0.4,
    "alert_threshold": 0.8,
    "processing_interval": 0.1
  },
  "system_config": {
    "max_file_size": 10485760,
    "supported_formats": ["jpeg", "png", "webp"],
    "recording_duration": 300
  }
}
```

### PUT /config
Update system configuration.

**Request Body:**
```json
{
  "ai_config": {
    "confidence_threshold": 0.5
  }
}
```

---

## 📡 WebSocket

### WebSocket Connection
Connect to `ws://localhost:8002/ws` for real-time updates.

**Messages:**
```json
{
  "type": "violation_detected",
  "data": {
    "violation_id": 1,
    "type": "no_helmet",
    "camera": "Main Entrance",
    "timestamp": "2026-03-27T17:20:00Z"
  }
}
```

```json
{
  "type": "alert_created",
  "data": {
    "alert_id": 1,
    "level": "medium",
    "message": "Worker without safety vest"
  }
}
```

---

## 🚫 Rate Limiting

- **Default**: 100 requests per minute per IP
- **AI Detection**: 10 requests per minute per IP
- **File Upload**: 5 requests per minute per IP

---

## 📝 Error Codes

| Code | Description |
|------|-------------|
| VALIDATION_ERROR | Invalid input data |
| NOT_FOUND | Resource not found |
| UNAUTHORIZED | Authentication required |
| FORBIDDEN | Insufficient permissions |
| RATE_LIMITED | Too many requests |
| INTERNAL_ERROR | Server error |
| AI_MODEL_ERROR | AI processing error |
| DATABASE_ERROR | Database operation error |

---

## 🧪 Testing

### Test Endpoints
- `GET /test/health` - Test health endpoint
- `POST /test/detection` - Test AI detection with sample image
- `GET /test/database` - Test database connection

---

**Last Updated**: March 2026  
**API Version**: v1.0.0  
**Base URL**: http://localhost:8002
