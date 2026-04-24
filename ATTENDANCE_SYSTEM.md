# Attendance System - Implementation Guide

## 📋 Overview

This guide covers the **Face Recognition Attendance System** for your construction site monitoring application.

## 🚀 Phase 1: Enrollment System (Complete)

The foundation has been built with:

### Files Created:

1. **`backend/config/attendance_models.py`** - Database models and Pydantic schemas
2. **`backend/config/face_recognition.py`** - Face detection, embedding, and matching utilities
3. **`backend/routers/attendance.py`** - API endpoints for enrollment and management

### Database Collections:

- `employees` - Employee records
- `employee_face_profiles` - Face enrollments with embeddings
- `cameras` - RTSP camera configuration
- `attendance_logs` - Daily attendance records
- `face_detection_events` - Recognition events (for Phase 3)

### API Endpoints (Phase 1):

#### Employee Management
```
POST   /api/attendance/employees              - Create employee
GET    /api/attendance/employees              - List employees
GET    /api/attendance/employees/{id}         - Get employee details
```

#### Face Enrollment
```
POST   /api/attendance/employees/{id}/enroll-face     - Upload and process face photo (MULTIPART)
GET    /api/attendance/employees/{id}/face-profiles   - Get all enrolled faces
```

#### Camera Management
```
POST   /api/attendance/cameras              - Register RTSP camera
GET    /api/attendance/cameras              - List cameras
```

#### Attendance
```
POST   /api/attendance/attendance/mark      - Mark attendance
GET    /api/attendance/attendance/today     - Get today's attendance
```

## 📸 How Face Enrollment Works

### Step 1: Employee Registration
```bash
curl -X POST http://localhost:8002/api/attendance/employees \
  -H "Content-Type: application/json" \
  -d '{
    "employee_code": "EMP001",
    "name": "John Doe",
    "phone": "9876543210",
    "department": "Construction",
    "site_id": "site_01",
    "status": "active"
  }'
```

### Step 2: Upload Face Photo
```bash
curl -X POST http://localhost:8002/api/attendance/employees/{employee_id}/enroll-face \
  -F "file=@face.jpg" \
  -F "is_primary=true"
```

**Photo Requirements:**
- ✅ Clear face, frontal view
- ✅ Good lighting (avoid shadows on face)
- ✅ No glasses/sunglasses (if possible)
- ✅ No mask
- ✅ Single face per image
- ✅ Image size: 500x500 to 2000x2000 pixels

### Step 3: What Happens During Upload

1. **Face Detection** - Detects face in image using InsightFace
2. **Face Validation** - Rejects if:
   - No face found
   - Multiple faces detected
   - Very low quality
3. **Face Quality Check** - Scores based on:
   - Face size (should be 3-15% of image)
   - Brightness (0.2-0.9 range)
   - Blur detection (Laplacian variance)
   - Detector confidence
4. **Face Alignment** - Aligns face to standard orientation
5. **Embedding Generation** - Creates 512D vector using InsightFace
6. **Normalization** - Normalizes embedding for consistency
7. **Storage** - Saves:
   - Original image
   - Cropped face (112x112)
   - Embedding vector
   - Quality score

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Update requirements.txt
Already includes:
- `insightface==0.7.3` - Face detection & recognition
- `scikit-image==0.23.2` - Image processing
- `pymongo==4.6.1` - MongoDB driver

### 3. Create Directories
```bash
mkdir -p data/images/enrollments
mkdir -p data/images/violations
```

### 4. MongoDB Collections
Collections are auto-created on first run with proper validation schemas.

## 🧠 Face Recognition Technology

### InsightFace Model
- **Detector**: RetinaFace (detects faces)
- **Recognizer**: ArcFace (generates embeddings)
- **Type**: 512-dimensional face embeddings
- **Similarity**: Cosine similarity (0-1 range)
- **Threshold**: Default 0.75 (configurable)

### Embedding Process
```
Input Image → Face Detection → Face Alignment → Vector Embedding → Storage
```

### Matching Process
```
Camera Frame → Detect Faces → Generate Embeddings → Compare with DB → Identify Employee
```

## 📊 Database Schema

### Employees Collection
```javascript
{
  _id: ObjectId,
  employee_code: "EMP001",
  name: "John Doe",
  phone: "9876543210",
  department: "Construction",
  site_id: "site_01",
  status: "active",
  created_at: Date
}
```

### Employee Face Profiles
```javascript
{
  _id: ObjectId,
  employee_id: ObjectId,
  photo_url: "/enrollment_files/employee_xxx.jpg",
  cropped_face_url: "/enrollment_files/crop_xxx.jpg",
  embedding: [0.123, 0.456, ...], // 512-dim vector
  quality_score: 0.85,  // 0-1
  is_primary: true,
  created_at: Date
}
```

### Cameras
```javascript
{
  _id: ObjectId,
  camera_name: "Gate Entry",
  rtsp_url: "rtsp://192.168.1.71:554/11",
  location_name: "Main Gate",
  site_id: "site_01",
  active: true,
  frame_interval_sec: 2,
  match_enabled: true,
  created_at: Date
}
```

### Attendance Logs
```javascript
{
  _id: ObjectId,
  employee_id: ObjectId,
  camera_id: ObjectId,
  date: "2026-04-15",
  check_in_time: "2026-04-15T10:30:00Z",
  check_out_time: "2026-04-15T19:30:00Z",
  source: "rtsp_face",
  confidence: 0.92,
  status: "active"
}
```

## 🎯 Next Phases

### Phase 2: Recognition Worker (In Progress)
- Separate Python service for RTSP stream processing
- Continuous frame sampling
- Real-time face detection and matching
- Event publishing to main API

### Phase 3: Attendance Rules Engine
- Business logic for check-in/check-out
- Duplicate prevention (10-min cooldown)
- Time window validation
- Site-specific rules

### Phase 4: Live Dashboard
- React components for enrollment UI
- Real-time attendance feed
- Camera monitoring
- Admin dashboard

## 🔐 Security Considerations

1. **Face Data Storage**
   - Store embeddings (not raw pixels where possible)
   - Encrypt sensitive data
   - Implement access controls

2. **RTSP Credentials**
   - Never commit credentials to git
   - Use environment variables
   - Rotate credentials regularly

3. **Image Retention**
   - Define retention policy (e.g., 30 days)
   - Auto-delete old enrollments
   - Archive for compliance

4. **Access Control**
   - Role-based access (admin, manager, view-only)
   - Audit logs for sensitive operations
   - Consent tracking for employees

## 📈 Accuracy Tips

For best face recognition accuracy:

1. **Camera Placement**
   - Place at entry/exit gates
   - Head-height mounting (1.5-2m)
   - Avoid counter-lighting
   - Good outdoor/indoor lighting

2. **Enrollment Quality**
   - 3-5 photos per employee
   - Front-facing, neutral expression
   - Different lighting conditions
   - Slight angle variations

3. **Environment**
   - Consistent lighting
   - Fixed camera angle
   - Limited congestion
   - Clear face zone

4. **Tuning**
   - Start with 0.75 confidence threshold
   - Adjust based on false positive/negative rates
   - Monitor accuracy metrics
   - Collect low-confidence matches for review

## 🚨 Common Issues

### Issue: Face detection fails
**Solutions:**
- Ensure good lighting
- Face should be 5-20% of image
- No extreme angles
- Clear eyes and facial features

### Issue: Low quality score
**Cause (Solution):**
- Poor lighting → Improve lighting
- Small face → Bring camera closer
- Blurry image → Use tripod, good camera
- Side profile → Use front-facing photo

### Issue: Embedding mismatch
**Solutions:**
- Re-enroll with better photos
- Check database corruption
- Verify embeddings are normalized
- Check similarity threshold

## 📝 Testing API in Postman

### 1. Create Employee
```
POST http://localhost:8002/api/attendance/employees
Content-Type: application/json

{
  "employee_code": "EMP001",
  "name": "John Doe",
  "phone": "9876543210",
  "department": "Construction",
  "site_id": "site_01"
}
```

### 2. Upload Face (Binary File)
```
POST http://localhost:8002/api/attendance/employees/{employee_id}/enroll-face
Content-Type: multipart/form-data

file: (select image file)
is_primary: true
```

### 3. Get Employee Faces
```
GET http://localhost:8002/api/attendance/employees/{employee_id}/face-profiles
```

### 4. Register Camera
```
POST http://localhost:8002/api/attendance/cameras
Content-Type: application/json

{
  "camera_name": "Gate Entry",
  "rtsp_url": "rtsp://192.168.1.71:554/11",
  "location_name": "Main Gate",
  "site_id": "site_01",
  "frame_interval_sec": 2,
  "match_enabled": true
}
```

## 🎓 Recommended Reading

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [OpenCV RTSP Capture](https://docs.opencv.org/master/d6/d9e/classcv_1_1VideoCapture.html)
- [Face Recognition Best Practices](https://en.wikipedia.org/wiki/Facial_recognition_system)
- [MongoDB Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/overview/)

## 📞 Support

For issues or questions:
1. Check logs: `backend/server.py` execution logs
2. Verify MongoDB connection
3. Test API endpoints with Postman
4. Check image quality requirements
5. Verify file permissions

---

**Status**: Phase 1 (Enrollment System) ✅ Complete
**Next**: Phase 2 (Recognition Worker) - Coming soon
