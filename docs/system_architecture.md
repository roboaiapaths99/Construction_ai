# System Architecture Documentation

## Overview
The AI Construction Safety System is a distributed real-time monitoring platform that combines computer vision, web technologies, and database management to ensure construction site safety compliance.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (React SPA)   │◄──►│   (FastAPI)     │◄──►│   (MySQL)       │
│                 │    │                 │    │                 │
│ • Dashboard     │    │ • REST API      │    │ • Workers       │
│ • Live Camera   │    │ • WebSocket     │    │ • Violations    │
│ • Alerts        │    │ • AI Processing │    │ • Alerts        │
│ • Analytics     │    │ • File Storage  │    │ • Cameras       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   AI Module     │
                       │   (YOLOv8)      │
                       │                 │
                       │ • Object        │
                       │   Detection     │
                       │ • Violation      │
                       │   Recognition   │
                       │ • Real-time     │
                       │   Processing    │
                       └─────────────────┘
```

## Component Architecture

### Frontend Architecture
```
src/
├── components/           # Reusable UI Components
│   ├── Layout.jsx       # App Layout & Navigation
│   ├── WebcamSimple.jsx # Camera Component
│   ├── StatCard.jsx     # Statistics Display
│   └── AlertBanner.jsx  # Alert Notifications
├── pages/               # Page Components
│   ├── Dashboard.jsx    # Main Dashboard
│   ├── Cameras.jsx      # Camera Management
│   ├── Violations.jsx   # Violation Tracking
│   ├── Workers.jsx      # Worker Management
│   ├── Alerts.jsx       # Alert System
│   └── Settings.jsx     # System Settings
├── hooks/               # Custom React Hooks
│   └── useApiStable.js  # API with Polling
├── api/                 # API Configuration
│   └── index.js         # Endpoints & Config
└── utils/               # Utility Functions
```

### Backend Architecture
```
backend/
├── server.py            # Main FastAPI Application
├── config/              # Configuration Management
│   ├── database.py      # Database Settings
│   ├── settings.py      # Application Settings
│   └── ai_config.py     # AI Model Configuration
├── routers/             # API Route Handlers
├── services/            # Business Logic
├── models/              # Database Models
├── utils/               # Utility Functions
└── archive/             # Legacy Files
```

### AI Module Architecture
```
ai/
├── models/              # Trained Models
│   └── yolov8n.pt      # YOLOv8 Nano Model
├── datasets/            # Training Data
├── training/            # Training Scripts
└── inference/           # Inference Logic
```

## Data Flow Architecture

### Real-time Detection Flow
```
Camera Feed → Frontend → Backend → AI Module → Database → Frontend
     │            │         │          │           │         │
  Video Stream   │   Base64   │   Object   │   Store   │   Real-time
  (Webcam)       │   Image    │ Detection │   Data    │   Updates
                 │           │           │           │
                 ▼           ▼           ▼           ▼
             WebSocket   REST API   YOLOv8    MySQL    Live UI
             Connection  Processing  Model    Storage   Updates
```

### Alert Generation Flow
```
AI Detection → Violation Analysis → Alert Generation → Notification
      │                │                   │              │
  Object          Rule-based          Severity        Real-time
  Detection       Evaluation          Classification  WebSocket
                     │                   │              │
                 Multiple              Alert          Frontend
                 Violations            Levels         Notification
```

## Database Schema Architecture

### Entity Relationship Diagram
```
Workers ──┐
    │     │
    │     ▼
    │  Violations ──┐
    │              │
    ▼              ▼
Alerts ◄─────── Cameras
```

### Table Structures

#### Workers Table
```sql
CREATE TABLE workers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(100),
    status ENUM('Active', 'Inactive', 'At Risk'),
    location VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### Violations Table
```sql
CREATE TABLE violations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    worker_id INT,
    camera_name VARCHAR(255),
    violation_type ENUM('No Helmet', 'No Vest', 'Multiple Violations'),
    confidence FLOAT,
    bbox_x INT, bbox_y INT, bbox_width INT, bbox_height INT,
    image_path VARCHAR(500),
    status ENUM('open', 'resolved', 'ignored') DEFAULT 'open',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (worker_id) REFERENCES workers(id)
);
```

#### Alerts Table
```sql
CREATE TABLE alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    message TEXT NOT NULL,
    level ENUM('low', 'medium', 'high') DEFAULT 'medium',
    violation_id INT,
    camera_name VARCHAR(255),
    status ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    FOREIGN KEY (violation_id) REFERENCES violations(id)
);
```

## Security Architecture

### Authentication Layer
```
Client Request → JWT Validation → Rate Limiting → API Access
       │               │               │              │
   Browser/        Token          Request        Protected
   Mobile          Validation     Throttling     Endpoints
   App
```

### Data Protection
- **Input Validation**: All user inputs sanitized
- **SQL Injection Prevention**: Parameterized queries
- **File Upload Security**: Type and size validation
- **CORS Configuration**: Restricted origins in production
- **Rate Limiting**: Prevent API abuse

## Performance Architecture

### Caching Strategy
```
Frontend        Backend         Database
    │              │                │
React Cache    In-Memory       Query Cache
    │              │                │
Component      API Results     Indexed Tables
Level          Level            Level
```

### Scalability Considerations
- **Horizontal Scaling**: Load balancer ready
- **Database Pooling**: Connection management
- **Async Processing**: Non-blocking I/O
- **WebSocket Scaling**: Connection management

## Monitoring & Logging Architecture

### Logging Structure
```
logs/
├── server.log         # Server operations
├── detection.log      # AI detection events
├── error.log          # Error tracking
└── performance.log    # Performance metrics
```

### Monitoring Metrics
- **System Health**: CPU, Memory, Disk usage
- **API Performance**: Response times, error rates
- **AI Model Performance**: Inference speed, accuracy
- **Database Performance**: Query times, connection pool

## Deployment Architecture

### Development Environment
```
Local Machine
├── Frontend (npm start)    :3000
├── Backend (uvicorn)       :8002
├── Database (MySQL)         :3306
└── File Storage (Local)
```

### Production Architecture (Recommended)
```
Load Balancer
    │
    ├── Frontend (Nginx/React Build)
    │
    ├── Backend (FastAPI + Gunicorn)
    │   ├── Instance 1
    │   ├── Instance 2
    │   └── Instance 3
    │
    ├── Database (MySQL Cluster)
    │   ├── Primary
    │   └── Replica
    │
    ├── File Storage (S3/MinIO)
    │
    └── Monitoring (Prometheus + Grafana)
```

## Technology Stack Rationale

### Frontend Technology Choices
- **React**: Component-based architecture, large ecosystem
- **Tailwind CSS**: Utility-first, rapid development
- **Axios**: Promise-based HTTP client
- **React Router**: Client-side routing

### Backend Technology Choices
- **FastAPI**: Modern Python framework, automatic documentation
- **MySQL**: Reliable relational database
- **YOLOv8**: State-of-the-art object detection
- **Uvicorn**: ASGI server, WebSocket support

### AI Technology Choices
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation
- **OpenCV**: Computer vision operations

## Integration Patterns

### API Integration
- **RESTful Design**: Standard HTTP methods
- **JSON Format**: Consistent data exchange
- **Error Handling**: Standardized error responses
- **Versioning**: API versioning strategy

### Real-time Communication
- **WebSocket**: Bidirectional communication
- **Event-driven**: Real-time updates
- **Connection Management**: Automatic reconnection

## Future Architecture Considerations

### Microservices Migration
```
Monolith → Microservices
    │
    ├── User Service
    ├── Detection Service
    ├── Alert Service
    ├── Analytics Service
    └── Notification Service
```

### Cloud Integration
- **Containerization**: Docker deployment
- **Orchestration**: Kubernetes management
- **Cloud Storage**: Scalable file storage
- **CDN**: Global content delivery

---

**Document Version**: 1.0.0  
**Last Updated**: March 2026  
**Architecture Version**: Current
