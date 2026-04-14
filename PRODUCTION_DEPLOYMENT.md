# Production Deployment Guide

## Overview

This guide covers deploying the AI Construction Safety System to production environments.

## Pre-Deployment Checklist

### Security
- [ ] Generate strong SECRET_KEY
- [ ] Set DEBUG=false
- [ ] Update CORS_ORIGINS to production domain
- [ ] Configure HTTPS/SSL certificates
- [ ] Set up firewall rules
- [ ] Enable authentication (if not already)
- [ ] Configure rate limiting
- [ ] Set up API key management (future enhancement)

### Database
- [ ] Set up MySQL/MariaDB server
- [ ] Create database user with limited permissions
- [ ] Configure database backups
- [ ] Test database connection
- [ ] Run migrations
- [ ] Set up monitoring

### Application
- [ ] Update all environment variables
- [ ] Test with production-like data
- [ ] Configure logging
- [ ] Set up error tracking (Sentry recommended)
- [ ] Configure health checks
- [ ] Set up monitoring and alerts

### Infrastructure
- [ ] Test load balancing setup
- [ ] Configure CDN (if applicable)
- [ ] Set up reverse proxy (Nginx recommended)
- [ ] Configure caching strategy
- [ ] Set up log aggregation
- [ ] Test auto-scaling (if applicable)

## Deployment Options

### Option 1: Docker (Recommended)

#### 1. Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "server:app"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      ENV: production
      DEBUG: "false"
      DB_HOST: mysql
      DB_USER: safety_user
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: safety_ai
      SECRET_KEY: ${SECRET_KEY}
      CORS_ORIGINS: "https://yourdomain.com"
    depends_on:
      - mysql
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  mysql:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: safety_ai
      MYSQL_USER: safety_user
      MYSQL_PASSWORD: ${DB_PASSWORD}
    volumes:
      - mysql_data:/var/lib/mysql
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    restart: unless-stopped

volumes:
  mysql_data:
```

#### 3. Deploy
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Traditional Server Deployment

#### 1. Install Dependencies
```bash
sudo apt-get update
sudo apt-get install python3.11 python3-pip mysql-server nginx

# Create virtual environment
python3 -m venv /opt/safety-system/venv
source /opt/safety-system/venv/bin/activate

# Install Python packages
pip install -r backend/requirements.txt
pip install gunicorn
```

#### 2. Create Systemd Service
```ini
[Unit]
Description=AI Construction Safety System
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/safety-system
Environment="PATH=/opt/safety-system/venv/bin"
ExecStart=/opt/safety-system/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --access-logfile /var/log/safety-system/access.log \
    --error-logfile /var/log/safety-system/error.log \
    backend.server:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3. Configure Nginx
```nginx
upstream safety_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/your-cert.crt;
    ssl_certificate_key /etc/ssl/private/your-key.key;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API proxy
    location /api/ {
        proxy_pass http://safety_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
    
    # Frontend
    location / {
        root /opt/safety-system/frontend/build;
        try_files $uri $uri/ /index.html;
    }
}
```

##### 4. Enable Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable safety-system
sudo systemctl start safety-system
sudo systemctl restart nginx
```

## Health Check

```bash
curl https://yourdomain.com/api/health
```

Expected response:
```json
{
    "status": "healthy",
    "database": "connected",
    "version": "1.0.0"
}
```

## Monitoring

### Key Metrics to Monitor
- API response time
- Database connection pool usage
- Error rate
- Memory usage
- CPU usage
- Disk space
- Database size

### Recommended Tools
- Prometheus + Grafana
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Sentry (error tracking)
- New Relic
- DataDog

### Setup Example (Prometheus)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'safety-system'
    static_configs:
      - targets: ['localhost:8000']
```

## Logging

All logs are stored in `logs/` directory:
- `app_YYYY-MM-DD.log` - Application logs
- `errors_YYYY-MM-DD.log` - Error logs

Configure log aggregation:
```bash
# Example: Ship logs to ELK
filebeat install
filebeat setup -e
filebeat run
```

## Database Backups

### Automated Backup Script
```bash
#!/bin/bash
BACKUP_DIR="/backups/mysql"
DATE=$(date +%Y%m%d_%H%M%S)

mysqldump -u safety_user -p${DB_PASSWORD} \
    --all-databases \
    --single-transaction \
    > $BACKUP_DIR/backup_$DATE.sql

# Compress
gzip $BACKUP_DIR/backup_$DATE.sql

# Keep only last 30 backups
find $BACKUP_DIR -name 'backup_*.sql.gz' -mtime +30 -delete
```

Add to crontab:
```bash
0 2 * * * /usr/local/bin/backup-mysql.sh
```

## Scaling

### Horizontal Scaling
1. Deploy multiple API instances
2. Use load balancer (Nginx, HAProxy, AWS LB)
3. Use shared storage for uploads
4. Use shared MySQL database

### Vertical Scaling
1. Increase server resources (CPU, RAM)
2. Optimize database queries
3. Implement caching (Redis)
4. Optimize image processing

## Security Updates

1. Monitor for security advisories
2. Keep dependencies updated
3. Run regular security audits
4. Enable automatic security patches
5. Rotate secrets regularly

## Rollback Plan

1. Keep previous Docker image tagged
2. Git history for code rollback
3. Database backup before updates
4. Test updates in staging first

Example rollback:
```bash
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d backend
```

## Maintenance

### Weekly
- Review error logs
- Check disk space
- Monitor performance metrics

### Monthly
- Update dependencies
- Review and optimize slow queries
- Test backup restoration
- Security audit

### Quarterly
- Performance optimization review
- Capacity planning
- Disaster recovery drill
- Security penetration testing

## Support

For production issues:
1. Check logs: `tail -f logs/errors_*.log`
2. Monitor metrics
3. Check database connectivity
4. Verify environment variables
5. Contact support team

## Additional Resources

- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/
- MySQL Best Practices: https://dev.mysql.com/doc/
- Docker Documentation: https://docs.docker.com/
- Nginx Documentation: https://nginx.org/en/docs/
