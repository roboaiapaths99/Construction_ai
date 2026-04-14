# Troubleshooting Guide

## Common Issues & Solutions

### Connection Issues

#### Problem: Backend cannot connect to MySQL

**Symptoms:**
```
ERROR: 2003 (HY000): Can't connect to MySQL server on 'localhost' (61)
```

**Solutions:**

1. **Check MySQL is running**
```bash
# Linux/Mac
sudo systemctl status mysql
ps aux | grep mysql

# Windows
tasklist | findstr mysql
Get-Service MySQL80
```

2. **Verify MySQL credentials**
```bash
mysql -u root -p
# Enter password from .env
```

3. **Check MySQL port**
```bash
# Default is 3306
netstat -an | grep 3306
```

4. **Update connection string in .env**
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=safety_user
DB_PASSWORD=your_password
DB_NAME=safety_ai
```

5. **Check firewall rules**
```bash
# Allow port 3306
sudo ufw allow 3306
```

---

#### Problem: Frontend cannot reach backend API

**Symptoms:**
```
Error: Network Error
TypeError: fetch failed
```

**Solutions:**

1. **Check backend is running**
```bash
# Backend should be on port 8000 or 8001
curl http://localhost:8000/health
```

2. **Verify API_URL in .env**
```
REACT_APP_API_URL=http://localhost:8000
```

3. **Check CORS configuration**
```python
# In backend server.py
CORS_ORIGINS = ["http://localhost:3000"]
```

4. **Verify firewall allows traffic**
```bash
sudo ufw allow 8000
sudo ufw allow 3000
```

5. **Check browser console (F12) for CORS errors**

---

### Port Already in Use

#### Problem: Port 3000 (frontend) already in use

**Symptoms:**
```
Error: listen EADDRINUSE :::3000
```

**Solutions:**

1. **Find process using port**
```bash
# Linux/Mac
lsof -i :3000

# Windows
netstat -ano | findstr :3000
```

2. **Kill the process**
```bash
# Linux/Mac
kill -9 <PID>

# Windows
taskkill /PID <PID> /F
```

3. **Use different port**
```bash
# Set PORT environment variable
PORT=3001 npm start
```

---

#### Problem: Port 8000/8001 (backend) already in use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**

1. **Find process using port**
```bash
# Linux/Mac
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

2. **Kill or restart process**
```bash
kill -9 <PID>
```

3. **Change port in server.py**
```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

### Database Issues

#### Problem: Database tables don't exist

**Symptoms:**
```
OperationalError: no such table: violations
```

**Solutions:**

1. **Check if database exists**
```bash
mysql -u root -p -e "SHOW DATABASES;"
```

2. **Create database**
```bash
mysql -u root -p
mysql> CREATE DATABASE safety_ai;
mysql> CREATE USER 'safety_user'@'localhost' IDENTIFIED BY 'password';
mysql> GRANT ALL PRIVILEGES ON safety_ai.* TO 'safety_user'@'localhost';
mysql> FLUSH PRIVILEGES;
```

3. **Create tables manually**
```bash
# Run initialization script if available
mysql -u safety_user -p safety_ai < init.sql
```

4. **Restart backend** (should auto-initialize)
```bash
python server.py
```

---

#### Problem: Slow database queries

**Symptoms:**
- API responses take >2 seconds
- High CPU usage
- Timeouts

**Solutions:**

1. **Check for missing indexes**
```sql
SHOW INDEX FROM violations;
SHOW INDEX FROM cameras;
SHOW INDEX FROM workers;
SHOW INDEX FROM alerts;
```

2. **Add indexes**
```sql
ALTER TABLE violations ADD INDEX idx_camera_date (camera_id, created_at);
ALTER TABLE alerts ADD INDEX idx_severity (severity);
ALTER TABLE violations ADD INDEX idx_status (status);
```

3. **Optimize queries**
```sql
-- Find slow queries
SHOW VARIABLES LIKE 'slow_query_log';
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;

-- Check slow log
SHOW BINARY LOGS;
```

4. **Check connection pool**
```python
# In environment.py, increase pool_size
DB_POOL_SIZE=20
DB_POOL_RECYCLE=1800
```

5. **Monitor connections**
```sql
SHOW PROCESSLIST;
```

---

### Memory & Performance Issues

#### Problem: Backend consumes too much memory

**Symptoms:**
- High memory usage
- System becomes slow
- Out of memory errors

**Solutions:**

1. **Check memory usage**
```bash
# Linux
top
free -h

# Windows
tasklist /v | findstr python
Get-Process python | Select-Object WorkingSet
```

2. **Limit memory usage**
```bash
# Run with memory limit (Linux)
python -m memory_profiler server.py
```

3. **Check for memory leaks**
```python
# Add to server.py
import psutil
import os

def log_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Call periodically
```

4. **Optimize AI model loading**
```python
# Load model once, cache it
MODEL_CACHE = None

def get_model():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = YOLO(MODEL_PATH)  # Expensive operation
    return MODEL_CACHE
```

---

#### Problem: Frontend bundle size too large

**Symptoms:**
- Slow load times
- High data usage on mobile

**Solutions:**

1. **Analyze bundle**
```bash
npm run build
npm install -g webpack-bundle-analyzer
webpack-bundle-analyzer build/static/js/main.*.js
```

2. **Remove unused dependencies**
```bash
npm audit
npm prune
```

3. **Enable gzip compression**
```bash
# In Nginx
gzip on;
gzip_types text/plain application/javascript text/css;
```

4. **Lazy load routes**
```javascript
import React, { lazy, Suspense } from 'react';

const Violations = lazy(() => import('./pages/Violations'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/violations" element={<Violations />} />
      </Routes>
    </Suspense>
  );
}
```

---

### Authentication Issues

#### Problem: Login fails with "Invalid credentials"

**Symptoms:**
- Cannot login
- 401 Unauthorized
- "Invalid username or password"

**Solutions:**

1. **Verify credentials**
```bash
# Check user exists in database
SELECT * FROM users WHERE email = 'admin@example.com';
```

2. **Reset password**
```python
# In backend
from config.auth import AuthManager
auth = AuthManager()
hash = auth.hash_password("newpassword123")
# Update in database
```

3. **Check database user table**
```sql
DESCRIBE users;
SELECT COUNT(*) FROM users;
```

---

#### Problem: JWT token expired

**Symptoms:**
- 401 Unauthorized after logout
- "Token has expired"

**Solutions:**

1. **Check token expiration time**
```python
# In environment.py
JWT_EXPIRE_MINUTES=1440  # 24 hours
```

2. **Refresh token (backend)**
```python
@app.post("/api/refresh")
def refresh_token(current_user: dict = Depends(get_current_user)):
    new_token = auth.create_access_token({"user_id": current_user["id"]})
    return {"access_token": new_token}
```

3. **Implement token refresh in frontend**
```javascript
// Axios interceptor
api.interceptors.response.use(
  response => response,
  async error => {
    if (error.response.status === 401) {
      const newToken = await refresh();
      localStorage.setItem('token', newToken);
      // Retry request
    }
    return Promise.reject(error);
  }
);
```

---

### API Issues

#### Problem: API returns 500 Internal Server Error

**Symptoms:**
```json
{
  "detail": "Internal Server Error"
}
```

**Solutions:**

1. **Check backend logs**
```bash
tail -f logs/errors_*.log
tail -f logs/app_*.log
```

2. **Run backend in debug mode**
```python
# Set DEBUG=true in .env
DEBUG=true
```

3. **Check stack trace**
```bash
# Backend console should show full error
```

4. **Verify request format**
```bash
# Test with curl
curl -X POST http://localhost:8000/api/violations \
  -H "Content-Type: application/json" \
  -d '{"violation_type": "NO_HARD_HAT", "camera_id": 1}'
```

---

#### Problem: API returns 400 Bad Request

**Symptoms:**
```json
{
  "detail": "Validation error"
}
```

**Solutions:**

1. **Check request format**
```python
# Verify JSON is valid
import json
json.loads(request_body)
```

2. **Validate required fields**
```bash
curl -X POST http://localhost:8000/api/violations \
  -H "Content-Type: application/json" \
  -d '{"violation_type": "NO_HARD_HAT"}'  # Missing camera_id
```

3. **Check field types**
```python
# camera_id should be integer, not string
{"violation_type": "NO_HARD_HAT", "camera_id": "1"}  # Wrong
{"violation_type": "NO_HARD_HAT", "camera_id": 1}   # Correct
```

4. **Review API documentation**
```bash
curl http://localhost:8000/docs
```

---

#### Problem: API requests are slow

**Symptoms:**
- API responses take >2 seconds
- Timeouts
- 504 Gateway Timeout

**Solutions:**

1. **Check database query performance** (see Database Issues)

2. **Enable API request logging**
```python
# In logging_config.py
app_logger.info(f"API Request: {method} {path} - {response_time}ms")
```

3. **Add caching**
```python
from fastapi_cache2 import FastAPICache2
from fastapi_cache2.backends.redis import RedisBackend

@app.get("/api/violations", response_model=PaginatedResponse)
@cached(expire=300)  # Cache for 5 minutes
def get_violations():
    pass
```

4. **Limit results**
```python
# Add pagination
@app.get("/api/violations")
def get_violations(skip: int = 0, limit: int = 10):
    return db.query(Violation).offset(skip).limit(limit).all()
```

---

### Docker Issues

#### Problem: Docker container won't start

**Symptoms:**
```
ERROR: Container exited with code 1
```

**Solutions:**

1. **Check container logs**
```bash
docker logs <container_id>
docker logs ai_construction_backend
```

2. **Run with interactive shell**
```bash
docker run -it --entrypoint /bin/bash <image_name>
```

3. **Check Docker image**
```bash
docker images
docker inspect <image_name>
```

4. **Rebuild image**
```bash
docker build -t ai_construction_backend .
docker-compose build --no-cache
```

---

#### Problem: Docker containers can't communicate

**Symptoms:**
- Frontend can't reach backend
- Backend can't reach MySQL

**Solutions:**

1. **Check Docker network**
```bash
docker network ls
docker network inspect bridge
```

2. **Use container name as hostname**
```yaml
# In docker-compose.yml
services:
  backend:
    container_name: backend
  db:
    container_name: db

# In backend, use:
DB_HOST=db  # Not localhost
```

3. **Verify DNS resolution**
```bash
docker exec <container> ping db
docker exec <container> nslookup backend
```

---

### Deployment Issues

#### Problem: SSL certificate not working

**Symptoms:**
- Mixed content warnings
- Browser shows "Not Secure"
- SSL certificate error

**Solutions:**

1. **Verify certificate files exist**
```bash
ls -la /etc/ssl/certs/
ls -la /etc/ssl/private/
```

2. **Check Nginx SSL configuration**
```nginx
ssl_certificate /path/to/cert.pem;
ssl_certificate_key /path/to/key.pem;
```

3. **Test SSL certificate**
```bash
openssl s_client -connect localhost:443
```

4. **Renew certificate**
```bash
# Using certbot
sudo certbot renew
```

---

#### Problem: High CPU usage in production

**Symptoms:**
- System slow
- 100% CPU usage
- Services unresponsive

**Solutions:**

1. **Identify service using CPU**
```bash
top -b -n 1 | head -20
ps aux --sort=-%cpu | head -5
```

2. **Check for infinite loops**
```bash
# Monitor process
strace -p <PID>
```

3. **Increase worker processes**
```bash
# In server startup
gunicorn -w 4 --threads 2 server:app
```

4. **Reduce model inference frequency**
```python
# Cache inference results
# Batch process images
# Use lower confidence threshold
```

---

### Emergency Recovery

#### Problem: Complete system failure

**Recovery Steps:**

1. **Stop all services**
```bash
docker-compose down
sudo systemctl stop all relevant services
```

2. **Restore from backup**
```bash
# Check available backups
ls -la backups/

# Restore database
mysql -u root -p safety_ai < backups/latest.sql
```

3. **Clear logs and caches**
```bash
rm -f logs/*.log*
rm -rf /tmp/cache/*
```

4. **Start services**
```bash
docker-compose up -d
```

5. **Verify health**
```bash
curl http://localhost:8000/health
curl http://localhost:3000
```

---

### Getting Help

#### Collect Diagnostic Information

```bash
#!/bin/bash

# System info
echo "=== System Info ==="
uname -a
free -h
df -h

# Services status
echo "=== Services Status ==="
ps aux | grep -E "node|python|mysql"

# Port availability
echo "=== Port Availability ==="
netstat -an | grep -E ":3000|:8000|:3306"

# Recent logs
echo "=== Recent Logs ==="
tail -50 logs/errors_*.log
tail -50 logs/app_*.log

# Docker status
echo "=== Docker Status ==="
docker ps
docker logs backend
docker logs frontend
```

Run this and share output when reporting issues.

---

**Last Updated**: January 15, 2024
**Maintained By**: DevOps Team
