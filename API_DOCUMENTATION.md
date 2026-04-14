# AI Construction Safety System - API Documentation

## Base URL

```
http://localhost:8000 (development)
https://yourdomain.com/api (production)
```

## Authentication

All endpoints (except `/health`) require authentication using JWT tokens.

### Getting a Token

Tokens are issued through the authentication system. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## Endpoints

### Health & Status

#### 1. Health Check
```
GET /health
```

Response (200):
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Incidents & Violations

#### 1. Create Incident
```
POST /api/incidents
Authorization: Bearer <token>
Content-Type: application/json
```

Request body:
```json
{
  "camera_name": "Camera A - Building 1",
  "violation_type": "No Hard Hat",
  "confidence": 0.85,
  "bbox_x": 100,
  "bbox_y": 150,
  "bbox_width": 50,
  "bbox_height": 80,
  "image_path": "violation_123.jpg",
  "persons": 2
}
```

Response (201):
```json
{
  "success": true,
  "message": "Incident created successfully",
  "data": {
    "id": 123,
    "camera_name": "Camera A - Building 1",
    "violation_type": "No Hard Hat",
    "confidence": 0.85,
    "timestamp": "2024-01-15T10:30:00Z",
    "status": "open"
  }
}
```

#### 2. Get All Violations
```
GET /api/violations?page=1&per_page=10&status=open
Authorization: Bearer <token>
```

Query Parameters:
- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 10, max: 100)
- `status` (string): Filter by status (open, resolved, investigating)

Response (200):
```json
{
  "success": true,
  "message": "Violations retrieved",
  "data": [
    {
      "id": 123,
      "camera_name": "Camera A - Building 1",
      "violation_type": "No Hard Hat",
      "confidence": 0.85,
      "timestamp": "2024-01-15T10:30:00Z",
      "status": "open",
      "persons": 2
    }
  ],
  "pagination": {
    "total": 45,
    "page": 1,
    "per_page": 10,
    "total_pages": 5
  }
}
```

#### 3. Get Violation by ID
```
GET /api/violations/{id}
Authorization: Bearer <token>
```

Response (200):
```json
{
  "success": true,
  "data": {
    "id": 123,
    "camera_name": "Camera A - Building 1",
    "violation_type": "No Hard Hat",
    "confidence": 0.85,
    "timestamp": "2024-01-15T10:30:00Z",
    "status": "open",
    "image_path": "violation_123.jpg",
    "persons": 2
  }
}
```

#### 4. Update Violation Status
```
PATCH /api/violations/{id}/status
Authorization: Bearer <token>
Content-Type: application/json
```

Request body:
```json
{
  "status": "resolved"
}
```

Valid statuses: `open`, `investigating`, `resolved`

Response (200):
```json
{
  "success": true,
  "message": "Violation status updated",
  "data": {
    "id": 123,
    "status": "resolved"
  }
}
```

---

### Cameras

#### 1. Create Camera
```
POST /api/cameras
Authorization: Bearer <token>
Content-Type: application/json
```

Request body:
```json
{
  "name": "Camera A - Building 1",
  "location": "Main entrance",
  "rtsp_url": "rtsp://camera.local:554/stream",
  "is_active": true
}
```

Response (201):
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Camera A - Building 1",
    "location": "Main entrance",
    "rtsp_url": "rtsp://camera.local:554/stream",
    "is_active": true
  }
}
```

#### 2. Get All Cameras
```
GET /api/cameras?is_active=true
Authorization: Bearer <token>
```

Query Parameters:
- `is_active` (boolean): Filter by active status

Response (200):
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "Camera A - Building 1",
      "location": "Main entrance",
      "is_active": true,
      "last_seen": "2024-01-15T10:30:00Z"
    }
  ]
}
```

---

### Workers

#### 1. Create Worker
```
POST /api/workers
Authorization: Bearer <token>
Content-Type: application/json
```

Request body:
```json
{
  "name": "John Doe",
  "employee_id": "EMP001",
  "department": "Construction",
  "contact": "+1-555-0123"
}
```

Response (201):
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "John Doe",
    "employee_id": "EMP001",
    "department": "Construction",
    "contact": "+1-555-0123",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

#### 2. Get All Workers
```
GET /api/workers
Authorization: Bearer <token>
```

Response (200):
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "name": "John Doe",
      "employee_id": "EMP001",
      "department": "Construction"
    }
  ]
}
```

---

### Alerts

#### 1. Get All Alerts
```
GET /api/alerts?severity=high&is_read=false
Authorization: Bearer <token>
```

Query Parameters:
- `severity` (string): Filter by severity (low, medium, high, critical)
- `is_read` (boolean): Filter by read status

Response (200):
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "title": "High-risk violation detected",
      "description": "Multiple workers without hard hats",
      "severity": "high",
      "created_at": "2024-01-15T10:30:00Z",
      "is_read": false
    }
  ]
}
```

#### 2. Mark Alert as Read
```
PATCH /api/alerts/{id}/read
Authorization: Bearer <token>
```

Response (200):
```json
{
  "success": true,
  "message": "Alert marked as read"
}
```

---

## Error Responses

### Standard Error Response
```json
{
  "success": false,
  "error": "ERROR_CODE",
  "message": "Human readable error message",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "abc123def456"
}
```

### HTTP Status Codes

- `200 OK` - Successful GET/PATCH request
- `201 Created` - Successful POST request
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict (duplicate)
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Example Error Responses

#### 400 - Bad Request
```json
{
  "success": false,
  "error": "VALIDATION_ERROR",
  "message": "Invalid violation type",
  "details": {
    "field": "violation_type",
    "value": "Unknown",
    "reason": "Must be one of: No Hard Hat, No Safety Vest, etc."
  }
}
```

#### 401 - Unauthorized
```json
{
  "success": false,
  "error": "AUTHENTICATION_ERROR",
  "message": "Invalid or expired token"
}
```

#### 404 - Not Found
```json
{
  "success": false,
  "error": "NOT_FOUND",
  "message": "Violation with id 999 not found"
}
```

#### 429 - Rate Limited
```json
{
  "success": false,
  "error": "RATE_LIMITED",
  "message": "Rate limit exceeded. Maximum 100 requests per minute.",
  "retry_after": 45
}
```

---

## Rate Limiting

- **Limit**: 100 requests per minute per IP
- **Headers**:
  - `X-RateLimit-Limit`: 100
  - `X-RateLimit-Remaining`: 87
  - `X-RateLimit-Reset`: Unix timestamp

---

## Request/Response Logging

All requests are logged with request IDs for tracking:

Request includes:
- Method and path
- Client IP
- Timestamp

Response includes:
- Status code
- Processing time
- Request ID (X-Request-ID header)

Example:
```
REQUEST [abc-123]: GET /api/violations
RESPONSE [abc-123]: 200 (0.145s)
```

---

## API Versioning

Future versions will be available at:
```
/api/v2/
```

Current version: v1

---

## Pagination

All list endpoints support pagination:

Query Parameters:
- `page` (integer, default: 1): Page number
- `per_page` (integer, default: 10, max: 100): Items per page

Response includes:
```json
{
  "pagination": {
    "total": 100,
    "page": 1,
    "per_page": 10,
    "total_pages": 10
  }
}
```

---

## Filtering & Sorting

Supported filters:
- `status`: Filter by status
- `severity`: Filter by severity
- `is_active`: Filter by active status
- `is_read`: Filter by read status

Example:
```
GET /api/violations?status=open&page=1&per_page=20
```

---

## Bulk Operations

To be implemented in v2:
- Bulk update violations
- Bulk delete alerts
- Batch incident creation

---

## WebSocket (Real-time Updates)

To be implemented:
```
ws://localhost:8000/ws/live-alerts
wss://yourdomain.com/api/ws/live-alerts (production)
```

---

## Terms of Use

- API is provided as-is
- No SLA for uptime
- Rate limits may be adjusted
- API changes will be backward compatible (v1.x)

---

## Support

For API support, contact: api-support@constructionsafety.ai
