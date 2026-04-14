# Production Readiness Roadmap - Step by Step

## Phase 1: Preparation & Assessment (30 minutes)
## Phase 2: Security Hardening (1-2 hours)
## Phase 3: Code Quality & Testing (1-2 hours)
## Phase 4: Infrastructure Setup (1-2 hours)
## Phase 5: Monitoring & Logging (1 hour)
## Phase 6: Final Validation (30 minutes)
## Phase 7: Deployment (1-2 hours)

---

# PHASE 1: PREPARATION & ASSESSMENT (30 minutes)

## Step 1.1: Verify Current System Status

### Check What's Running
```bash
# Terminal 1: Check backend
curl http://localhost:8000/health

# Terminal 2: Check frontend
curl http://localhost:3000

# Terminal 3: Check database
mysql -u safety_user -p safety_ai -e "SELECT COUNT(*) FROM cameras;"
```

**Expected Results:**
- ✅ Backend responds with health status
- ✅ Frontend returns HTML
- ✅ Database is connected and has tables

---

## Step 1.2: Review Current State

**Backend Status:**
- [ ] Server running on port 8000
- [ ] MySQL database initialized
- [ ] All tables created (users, cameras, violations, etc.)
- [ ] API endpoints responding

**Frontend Status:**
- [ ] React app running on port 3000
- [ ] All pages loading without errors
- [ ] Components rendering correctly

**Configuration Status:**
- [ ] Environment variables configured
- [ ] .env file exists with correct values
- [ ] No hardcoded secrets in code

---

## Step 1.3: Document Current Environment

Create `DEPLOYMENT_INFO.md`:

```markdown
# Current Environment Info

## Backend
- Python Version: 3.11+
- Framework: FastAPI
- Current Port: 8000
- Database: MySQL (safety_ai)
- Requirements: backend/requirements.txt

## Frontend
- Node Version: 18+
- Framework: React 18.2
- Current Port: 3000
- Package File: frontend/package.json

## Database
- Host: localhost
- Port: 3306
- Database: safety_ai
- User: safety_user

## Directory Structure
```
ai_construction_system/
├── backend/                 # FastAPI application
├── frontend/               # React application
├── ai/                    # AI models and inference
├── data/                  # Data storage
├── logs/                  # Application logs
├── docs/                  # Documentation
├── docker-compose.yml     # Docker configuration
└── .env                  # Environment variables
```

## Documentation
- PRODUCTION_README.md
- API_DOCUMENTATION.md
- PRODUCTION_DEPLOYMENT.md
- SECURITY_HARDENING.md
- DATABASE_OPERATIONS.md
- TESTING.md
- TROUBLESHOOTING.md
```

---

# PHASE 2: SECURITY HARDENING (1-2 hours)

## Step 2.1: Environment Configuration

### Update .env with Production Values

```bash
# In your .env file, ensure these are set:

# SECURITY
ENV=production
DEBUG=false
SECRET_KEY=<generate-new-strong-key>  # Use: python -c "import secrets; print(secrets.token_urlsafe(32))"

# DATABASE
DB_HOST=localhost
DB_PORT=3306
DB_USER=safety_user
DB_PASSWORD=<strong-password>
DB_NAME=safety_ai
DB_POOL_SIZE=10
DB_POOL_RECYCLE=1800

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
CORS_CREDENTIALS=true

# JWT
JWT_SECRET_KEY=<same-as-SECRET_KEY>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# RATE LIMITING
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# LOGGING
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FORMAT=json

# AI MODEL
YOLO_MODEL_PATH=ai/models/yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.5
YOLO_IOU_THRESHOLD=0.45

# FILE UPLOAD
MAX_UPLOAD_SIZE=52428800  # 50MB
UPLOAD_DIR=data/uploads
ALLOWED_EXTENSIONS=['jpg', 'jpeg', 'png', 'mp4', 'avi']
```

### Verification Checklist ✅
- [ ] DEBUG is set to `false`
- [ ] SECRET_KEY is randomly generated (32+ chars)
- [ ] All database credentials set correctly
- [ ] CORS_ORIGINS configured for your domain
- [ ] No sensitive data in git (add .env to .gitignore)

---

## Step 2.2: Database Security

### Change MySQL Passwords

```bash
# Login to MySQL
mysql -u root -p

# Change root password
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new-strong-password';

# Change safety_user password
ALTER USER 'safety_user'@'localhost' IDENTIFIED BY 'new-strong-password';

# Apply changes
FLUSH PRIVILEGES;

# Exit
EXIT;
```

### Disable Remote MySQL Access

```bash
# Edit MySQL config
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf

# Find and verify this line (should be present):
# bind-address = 127.0.0.1

# Restart MySQL
sudo systemctl restart mysql
```

### Remove Test Accounts

```bash
mysql -u root -p -e "
USE mysql;
DELETE FROM user WHERE user='test' OR user='guest';
FLUSH PRIVILEGES;
"
```

---

## Step 2.3: Generate Production Credentials

### Create Strong Keys

```bash
# Generate SECRET_KEY
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate database password
python -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(16))"
```

**SAVE THESE SECURELY** (not in git, use secure password manager)

---

## Step 2.4: Enable HTTPS/SSL

### Generate Self-Signed Certificate (for testing)

```bash
# Create certificates directory
mkdir -p certs

# Generate self-signed certificate (valid 365 days)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out certs/cert.pem \
  -keyout certs/key.pem \
  -days 365 \
  -subj "/CN=localhost"

# Set permissions
chmod 600 certs/key.pem
chmod 644 certs/cert.pem
```

### For Production (Get Real Certificate)

Option 1: Using Let's Encrypt (FREE)
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot certonly --nginx -d yourdomain.com

# Auto-renew
sudo systemctl enable certbot.timer
```

Option 2: Using Self-Signed
```bash
# For production, use proper certificate authority
# Request certificate from CA or use AWS Certificate Manager
```

---

# PHASE 3: CODE QUALITY & TESTING (1-2 hours)

## Step 3.1: Code Quality Checks

### Backend Code Quality

```bash
# Navigate to backend
cd backend

# Install testing dependencies
pip install pytest pytest-cov bandit black flake8

# Run security audit
bandit -r . -ll  # Only show major issues

# Format code
black .

# Check linting
flake8 . --max-line-length=100

# Fix issues automatically where possible
black . --line-length=100
```

### Frontend Code Quality

```bash
# Navigate to frontend
cd frontend

# Install lint tools
npm install --save-dev eslint-config-prettier prettier

# Format code
npx prettier --write src/

# Run ESLint
npx eslint src/

# Fix issues
npx eslint src/ --fix
```

---

## Step 3.2: Run Test Suite

### Backend Tests

```bash
cd backend

# Run all tests
pytest -v --cov=. --cov-report=html

# Check coverage
# Open htmlcov/index.html to view coverage report

# Run only critical tests
pytest -v -m critical
```

### Frontend Tests

```bash
cd frontend

# Run tests
npm test -- --coverage

# Watch mode for development
npm test -- --watch
```

---

## Step 3.3: Performance Testing

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test single request
ab -n 1 http://localhost:8000/api/violations

# Load test: 100 requests, 10 concurrent
ab -n 100 -c 10 http://localhost:8000/api/violations

# Heavy load: 1000 requests, 50 concurrent
ab -n 1000 -c 50 http://localhost:8000/api/violations

# Expected results:
# - Response time: < 200ms
# - Throughput: > 50 req/sec
# - Failed requests: < 1%
```

---

# PHASE 4: INFRASTRUCTURE SETUP (1-2 hours)

## Step 4.1: Set Up Logging

### Create Log Configuration

```bash
# Create logs directory
mkdir -p logs

# Create log rotation config
sudo nano /etc/logrotate.d/ai-construction

# Add:
/path/to/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 appuser appuser
}
```

### Verify Logging Works

```bash
# Check if logs are being created
ls -lah logs/

# Monitor logs in real-time
tail -f logs/app_*.log
```

---

## Step 4.2: Docker Setup

### Build Docker Images

```bash
# Build backend image
docker build -t ai-construction-backend backend/

# Build frontend image
docker build -t ai-construction-frontend frontend/

# Verify images
docker images | grep ai-construction

# Test backend image
docker run -it --rm \
  -e DB_HOST=host.docker.internal \
  -e DEBUG=false \
  ai-construction-backend

# Test frontend image
docker run -it --rm -p 3000:3000 ai-construction-frontend
```

### Create docker-compose.prod.yml

```yaml
version: '3.8'

services:
  database:
    image: mysql:8.0
    container_name: ai_db_prod
    environment:
      MYSQL_ROOT_PASSWORD: ${DB_ROOT_PASSWORD}
      MYSQL_DATABASE: ${DB_NAME}
      MYSQL_USER: ${DB_USER}
      MYSQL_PASSWORD: ${DB_PASSWORD}
    volumes:
      - db-data:/var/lib/mysql
    ports:
      - "3306:3306"
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always

  backend:
    build: ./backend
    container_name: ai_backend_prod
    environment:
      ENV: production
      DEBUG: "false"
      DB_HOST: database
      DB_PORT: 3306
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      SECRET_KEY: ${SECRET_KEY}
    ports:
      - "8000:8000"
    depends_on:
      database:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always

  frontend:
    build: ./frontend
    container_name: ai_frontend_prod
    environment:
      REACT_APP_API_URL: http://localhost:8000
      REACT_APP_ENV: production
    ports:
      - "3000:3000"
    depends_on:
      - backend
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: always

volumes:
  db-data:
```

### Test Docker Compose

```bash
# Create production .env
cp .env .env.prod
# Edit .env.prod with production values

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify services
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:3000
```

---

## Step 4.3: Set Up Nginx Reverse Proxy

### Create Nginx Configuration

```bash
# Create config directory
sudo mkdir -p /etc/nginx/conf.d

# Create AI Construction config
sudo nano /etc/nginx/conf.d/ai-construction.conf
```

Add this content:

```nginx
# Upstream services
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:3000;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    gzip_min_length 1000;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Disable buffering for streaming
        proxy_buffering off;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://backend;
        access_log off;
    }
}
```

### Validate and Start Nginx

```bash
# Test config syntax
sudo nginx -t

# Start Nginx
sudo systemctl start nginx

# Enable on boot
sudo systemctl enable nginx

# Check status
sudo systemctl status nginx

# View logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

# PHASE 5: MONITORING & LOGGING (1 hour)

## Step 5.1: Set Up Application Monitoring

### Configure Application Monitoring

```python
# Add to backend/server.py

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Initialize Sentry (error tracking)
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment=os.getenv("ENV", "production")
)

# Or use simple file-based logging (already implemented in logging_config.py)
```

### Add Basic Monitoring Script

```bash
# Create monitoring script
cat > scripts/monitor.sh << 'EOF'
#!/bin/bash

HEALTH_URL="http://localhost:8000/health"
LOG_FILE="logs/monitoring.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check backend health
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
    
    if [ $RESPONSE -eq 200 ]; then
        echo "$TIMESTAMP - Backend: OK" >> $LOG_FILE
    else
        echo "$TIMESTAMP - Backend: ERROR ($RESPONSE)" >> $LOG_FILE
        # Send alert
        echo "Backend down!" | mail -s "ALERT" admin@example.com
    fi
    
    # Check CPU usage
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "$TIMESTAMP - CPU: ${CPU}%" >> $LOG_FILE
    
    # Check disk usage
    DISK=$(df -h / | awk 'NR==2 {print $5}' | cut -d'%' -f1)
    echo "$TIMESTAMP - Disk: ${DISK}%" >> $LOG_FILE
    
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x scripts/monitor.sh
```

### Start Monitoring

```bash
# Run in background
nohup ./scripts/monitor.sh > logs/monitor.log 2>&1 &

# Or use systemd
sudo systemctl start ai-construction-monitor
```

---

## Step 5.2: Set Up Log Aggregation

### Configure Centralized Logging

```bash
# View logs from all services
tail -f logs/app_*.log logs/errors_*.log

# Search logs for errors
grep -r "ERROR" logs/

# Monitor specific endpoint
tail -f logs/*.log | grep "/api/violations"
```

---

# PHASE 6: FINAL VALIDATION (30 minutes)

## Step 6.1: Pre-Production Checklist

Run through this complete checklist:

### Security ✅
- [ ] DEBUG mode disabled
- [ ] SECRET_KEY is strong and unique
- [ ] Database passwords changed
- [ ] HTTPS/SSL configured
- [ ] CORS origins restricted
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] All dependencies updated

### Code Quality ✅
- [ ] Bandit security scan passed
- [ ] ESLint warnings fixed
- [ ] Tests passing (>80% coverage)
- [ ] No console.logs in production code
- [ ] Error messages don't leak info
- [ ] All TODOs resolved

### Performance ✅
- [ ] API response time < 200ms
- [ ] Frontend bundle < 500KB
- [ ] Database queries optimized
- [ ] Caching configured
- [ ] Load test passed (1000 concurrent)

### Infrastructure ✅
- [ ] Docker images built and tested
- [ ] Nginx configured correctly
- [ ] SSL certificate valid
- [ ] Logging working
- [ ] Monitoring active
- [ ] Backup procedures in place
- [ ] Firewall configured

### Data ✅
- [ ] Database backups scheduled
- [ ] Sensitive data encrypted
- [ ] GDPR compliance checked
- [ ] Data retention policy set
- [ ] Sample data loaded

---

## Step 6.2: End-to-End Testing

### Test Complete User Journey

```bash
# 1. Test authentication (if implemented)
curl -X POST http://localhost:8000/api/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "password"}'

# 2. Test violations API
curl http://localhost:8000/api/violations

# 3. Test camera API
curl http://localhost:8000/api/cameras

# 4. Test health check
curl http://localhost:8000/health

# 5. Test frontend
open http://localhost:3000

# 6. Test with HTTPS (if configured)
curl https://yourdomain.com/health
```

---

## Step 6.3: Verify All Systems

```bash
# Check all services running
ps aux | grep -E "python|node|mysql|nginx"

# Check ports
netstat -tulpn | grep LISTEN

# Check logs for errors
grep -i error logs/*.log

# Check disk space
df -h

# Check memory usage
free -h
```

---

# PHASE 7: DEPLOYMENT (1-2 hours)

## Step 7.1: Backup Current State

```bash
# Database backup
mysqldump -u root -p --all-databases | gzip > backup_$(date +%Y%m%d).sql.gz

# Application backup
tar -czf app_backup_$(date +%Y%m%d).tar.gz . --exclude=.git --exclude=node_modules --exclude=.venv

# Verify backups
ls -lh *.gz
```

---

## Step 7.2: Production Deployment

### Option A: Docker Compose Deployment

```bash
# Pull latest code
git pull origin main

# Stop current services
docker-compose -f docker-compose.prod.yml down

# Build new images
docker-compose -f docker-compose.prod.yml build

# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Verify services
docker-compose -f docker-compose.prod.yml ps

# Check health
curl https://yourdomain.com/health
```

### Option B: Traditional Server Deployment

```bash
# 1. Update backend
cd backend
git pull
pip install -r requirements.txt
# Restart backend service
sudo systemctl restart ai-construction-backend

# 2. Update frontend
cd ../frontend
git pull
npm install
npm run build
# Files in build/ are served by Nginx
sudo systemctl restart nginx

# 3. Verify
curl https://yourdomain.com/health
```

---

## Step 7.3: Post-Deployment Verification

```bash
# Check all endpoints respond
curl https://yourdomain.com/
curl https://yourdomain.com/api/violations
curl https://yourdomain.com/health

# Check logs for errors
tail -f logs/errors_*.log

# Monitor performance
top
df -h

# Verify database
mysql -u safety_user -p safety_ai -e "SELECT COUNT(*) FROM violations;"
```

---

# Next Steps After Deployment

## Weekly Tasks
- [ ] Review error logs
- [ ] Check disk usage
- [ ] Verify backups completed
- [ ] Monitor performance metrics

## Monthly Tasks
- [ ] Update dependencies
- [ ] Review security
- [ ] Optimize slow queries
- [ ] Archive old data

## Every 90 Days
- [ ] Full security audit
- [ ] Test disaster recovery
- [ ] Load testing
- [ ] Performance optimization

---

## Support Resources

- **Documentation**: See DOCUMENTATION_INDEX.md
- **Troubleshooting**: See TROUBLESHOOTING.md
- **Security**: See SECURITY_HARDENING.md
- **API**: See API_DOCUMENTATION.md

---

**Status**: Step-by-step production readiness guide complete!
**Next**: Follow Phase 1 to get started →
