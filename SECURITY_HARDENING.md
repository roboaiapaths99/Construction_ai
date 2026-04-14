# Security Hardening Guide

## Security Framework

This guide provides comprehensive security hardening procedures for production deployment.

## Pre-Production Security Checklist

### Authentication & Authorization
- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY (32+ characters, random)
- [ ] Enable JWT token expiration
- [ ] Implement password complexity requirements (min 12 chars, mixed case, numbers, symbols)
- [ ] Setup Multi-Factor Authentication (MFA) for admin accounts
- [ ] Audit all user roles and permissions
- [ ] Remove test/demo user accounts
- [ ] Enable account lockout after failed login attempts (5 attempts, 30 min lockout)

**Generate Secure SECRET_KEY:**
```python
import secrets
print(secrets.token_urlsafe(32))
# Output: aBcDefGhIjKlMnOpQrStUvWxYz0123456789
```

### Network Security
- [ ] Enable HTTPS/TLS with valid certificate
- [ ] Disable HTTP (redirect to HTTPS)
- [ ] Enable HSTS (Strict-Transport-Security for 1 year)
- [ ] Configure CORS to only trusted origins
- [ ] Enable firewall (UFW on Linux)
- [ ] Restrict SSH access to known IPs
- [ ] Close unnecessary ports
- [ ] Setup rate limiting (100 req/min per IP)
- [ ] Enable DDoS protection

**Firewall Configuration (UFW):**
```bash
# Enable firewall
sudo ufw enable

# Allow SSH (first! don't lock yourself out)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block all others by default
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Verify
sudo ufw status verbose
```

### Database Security
- [ ] Change MySQL root password
- [ ] Remove anonymous MySQL users
- [ ] Disable remote MySQL access (bind to localhost)
- [ ] Create dedicated DB user with limited privileges
- [ ] Enable SSL for database connections
- [ ] Encrypt database backups
- [ ] Restrict database file permissions (600)
- [ ] Enable query logging for audits
- [ ] Implement connection limits

**Secure MySQL Setup:**
```bash
# Run security script
sudo mysql_secure_installation

# Verify secure settings
mysql -u root -p
mysql> SELECT user, host, authentication_string FROM mysql.user;
mysql> SHOW VARIABLES LIKE 'require_secure_transport';
```

### Data Protection
- [ ] Enable encryption at rest for sensitive data
- [ ] Encrypt database backups
- [ ] Enable encryption in transit (TLS)
- [ ] Hash all passwords (bcrypt)
- [ ] Use environment variables for secrets (not hardcoded)
- [ ] Implement data expiration policies
- [ ] Anonymize old data
- [ ] Enable audit logging
- [ ] Classify data by sensitivity

**Environment Variables Security:**
```bash
# .env file permissions (Linux/Mac)
chmod 600 .env

# Never commit .env to git
echo ".env" >> .gitignore

# Use .env.example for documentation
cp .env .env.example
# Remove sensitive values from .env.example
```

### API Security
- [ ] Implement input validation (reject malformed data)
- [ ] Sanitize all user input
- [ ] Use parameterized queries (prevent SQL injection)
- [ ] Implement CSRF tokens
- [ ] Add security headers (CSP, X-Frame-Options, etc.)
- [ ] Rate limit API endpoints
- [ ] Implement request signing
- [ ] Use API versioning
- [ ] Deprecate old API versions
- [ ] Monitor unusual API usage patterns

**Security Headers Configuration (Nginx):**
```nginx
# HTTP Security Headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

### Application Security
- [ ] Update all dependencies (pip, npm)
- [ ] Run security audits (bandit for Python, npm audit)
- [ ] Implement error handling (no stack traces in production)
- [ ] Disable debug mode
- [ ] Enable logging and monitoring
- [ ] Implement circuit breaker patterns
- [ ] Use dependency scanning tools
- [ ] Review and fix security warnings
- [ ] Implement secrets management

**Security Audits:**
```bash
# Python
pip install bandit
bandit -r backend/

# Node.js
npm audit
npm audit fix

# OWASP Dependency Check
curl https://jeremylong.github.io/DependencyCheck/release/latest | grep -oP 'href="\K[^"]*\.zip'
```

### Infrastructure Security
- [ ] Keep OS updated (security patches)
- [ ] Enable SELinux or AppArmor
- [ ] Implement container security scanning
- [ ] Use non-root users in containers
- [ ] Implement network isolation
- [ ] Use secrets vaults (HashiCorp Vault, AWS Secrets Manager)
- [ ] Enable audit logging
- [ ] Backup configurations
- [ ] Document infrastructure

**Container Security:**
```dockerfile
# In Dockerfile
# Use specific image versions (not latest)
FROM python:3.11-slim

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Read-only file system where possible
RUN chmod -R 555 /app

# No sensitive data in images
ARG SECRET_KEY
# Use BuildKit secrets instead
```

### Monitoring & Logging
- [ ] Centralized logging (ELK, Splunk, CloudWatch)
- [ ] Log aggregation
- [ ] Real-time alerting
- [ ] Security event monitoring
- [ ] Failed login tracking
- [ ] Unauthorized access attempts
- [ ] API abuse patterns
- [ ] Database access logs
- [ ] File integrity monitoring

**Logging Configuration:**
```python
# In logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import json

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
        handler = RotatingFileHandler('logs/security.log', maxBytes=10*1024*1024, backupCount=5)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)
    
    def log_failed_login(self, username, ip):
        self.logger.warning(f"FAILED_LOGIN: {username} from {ip}")
    
    def log_unauthorized_access(self, user_id, endpoint, ip):
        self.logger.warning(f"UNAUTHORIZED: user={user_id} endpoint={endpoint} ip={ip}")
    
    def log_api_abuse(self, ip, requests_count):
        self.logger.error(f"API_ABUSE: ip={ip} requests={requests_count}")

security_logger = SecurityLogger()
```

### Incident Response
- [ ] Have incident response plan
- [ ] Document security contacts
- [ ] Backup and recovery procedures
- [ ] Communication plan
- [ ] Post-incident review process
- [ ] Regular disaster recovery drills

## Runtime Security

### Secrets Management

**Using Environment Variables:**
```bash
# Load from secure key management system
export SECRET_KEY=$(aws secretsmanager get-secret-value --secret-id api-secret-key --query SecretString --output text)
export DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id db-password --query SecretString --output text)

# Start application
python server.py
```

**Using HashiCorp Vault:**
```python
import hvac

client = hvac.Client(url='http://127.0.0.1:8200')
secret = client.secrets.kv.v2.read_secret_version(path='database/config')
db_password = secret['data']['data']['password']
```

### Rate Limiting Implementation

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute=100):
        self.requests_per_minute = requests_per_minute
        self.clients = defaultdict(list)
    
    def is_rate_limited(self, client_id):
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if req_time > minute_ago
        ]
        
        # Check if over limit
        if len(self.clients[client_id]) >= self.requests_per_minute:
            return True
        
        # Record this request
        self.clients[client_id].append(now)
        return False

# Usage in FastAPI
rate_limiter = RateLimiter(requests_per_minute=100)

@app.get("/api/violations")
async def get_violations(request: Request):
    client_ip = request.client.host
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # ... rest of endpoint
```

### Input Validation

```python
from pydantic import BaseModel, validator, EmailStr
import re

class IncidentCreate(BaseModel):
    violation_type: str
    description: str
    camera_id: int
    
    @validator('violation_type')
    def validate_type(cls, v):
        allowed = ['NO_HARD_HAT', 'NO_SAFETY_VEST', 'UNSAFE_POSTURE']
        if v not in allowed:
            raise ValueError(f'Invalid violation type: {v}')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if len(v) > 1000:
            raise ValueError('Description too long')
        # Prevent HTML/JavaScript
        if any(tag in v for tag in ['<script>', '<iframe>', 'javascript:']):
            raise ValueError('Invalid content in description')
        return v
    
    @validator('camera_id')
    def validate_camera(cls, v):
        if v < 1:
            raise ValueError('Invalid camera ID')
        return v
```

### SQL Injection Prevention

```python
# WRONG - vulnerable to SQL injection
query = f"SELECT * FROM violations WHERE camera_id = {camera_id}"
db.execute(query)

# CORRECT - parameterized query
query = "SELECT * FROM violations WHERE camera_id = %s"
db.execute(query, (camera_id,))

# In SQLAlchemy (ORM)
violations = db.query(Violation).filter(Violation.camera_id == camera_id).all()
```

### XSS Prevention

```python
from html import escape

# Sanitize user input
user_input = "<script>alert('xss')</script>"
sanitized = escape(user_input)
# Result: &lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;

# In frontend (React)
# Automatically escapes by default
<div>{userInput}</div>

# For HTML content, use DOMPurify
import DOMPurify from 'dompurify';
const clean = DOMPurify.sanitize(userHTML);
```

## Security Monitoring

### Key Events to Monitor

```python
class SecurityMonitor:
    def __init__(self):
        self.alerts = []
    
    def check_failed_logins(self, user_id):
        """Alert on multiple failed login attempts"""
        failed_count = get_failed_login_count(user_id, minutes=30)
        if failed_count > 5:
            self.alert(f"Multiple failed logins for {user_id}")
    
    def check_unusual_api_patterns(self, ip):
        """Alert on suspicious API usage"""
        req_count = get_recent_requests(ip, minutes=1)
        if req_count > 200:
            self.alert(f"High API usage from {ip}: {req_count} req/min")
    
    def check_unauthorized_access(self, user_id, resource):
        """Alert on unauthorized access attempts"""
        self.alert(f"Unauthorized access by {user_id} to {resource}")
    
    def alert(self, message):
        """Send security alert"""
        # Log to security log
        security_logger.log_critical(message)
        # Send to monitoring system
        send_to_sentry(message)
        # Send to admin
        send_email_alert(message)
```

### Security Dashboard

Create Grafana dashboard monitoring:
- Failed login attempts
- API rate limit violations
- Unauthorized access attempts
- Database connection errors
- SSL certificate expiration
- Firewall rule violations

## Compliance

### GDPR Compliance

```python
# User data deletion
@app.delete("/api/users/{user_id}/data")
async def delete_user_data(user_id: int, current_user = Depends(get_current_user)):
    """GDPR data deletion right"""
    if current_user.id != user_id and current_user.role != 'admin':
        raise HTTPException(status_code=403)
    
    # Delete user data
    db.query(Violation).filter(Violation.user_id == user_id).delete()
    db.query(Alert).filter(Alert.assigned_to == user_id).delete()
    db.query(User).filter(User.id == user_id).delete()
    db.commit()
    
    # Log deletion for compliance
    audit_log.log(f"User {user_id} data deleted by {current_user.id}")
```

### Data Retention Policy

```python
# Archive old data
@app.post("/api/admin/archive")
async def archive_old_data(current_user = Depends(get_current_user)):
    """Archive violations older than 1 year"""
    if current_user.role != 'admin':
        raise HTTPException(status_code=403)
    
    retention_date = datetime.now() - timedelta(days=365)
    old_violations = db.query(Violation).filter(
        Violation.created_at < retention_date
    ).all()
    
    # Export to archive
    export_to_archive(old_violations)
    
    # Delete from main database
    db.query(Violation).filter(
        Violation.created_at < retention_date
    ).delete()
    db.commit()
```

## Penetration Testing

### Common Vulnerabilities to Test

1. **Authentication Bypass**
```bash
# Test without token
curl http://localhost:8000/api/violations

# Test with invalid token
curl -H "Authorization: Bearer invalid" http://localhost:8000/api/violations

# Test expired token
curl -H "Authorization: Bearer expired_token" http://localhost:8000/api/violations
```

2. **SQL Injection**
```bash
curl "http://localhost:8000/api/violations?camera_id=1 OR 1=1"
curl "http://localhost:8000/api/violations?camera_id=1; DROP TABLE violations"
```

3. **XSS**
```bash
curl -X POST http://localhost:8000/api/violations \
  -H "Content-Type: application/json" \
  -d '{"description": "<script>alert(1)</script>"}'
```

4. **CSRF**
```bash
# Test CSRF token requirement
curl -X POST http://localhost:8000/api/violations \
  -H "Content-Type: application/json" \
  -d '{...}' \
  # Missing CSRF token
```

## Security Incident Response

### If Breach Occurs

1. **Immediate Actions**
   - Isolate affected systems
   - Preserve logs and evidence
   - Notify security team
   - Don't shut down immediately

2. **Assessment**
   - Determine scope of breach
   - Identify compromised data
   - Review access logs
   - Check for lateral movement

3. **Containment**
   - Revoke compromised credentials
   - Patch vulnerabilities
   - Update security rules
   - Reset passwords

4. **Notification**
   - Notify affected users
   - Inform regulatory bodies (if required)
   - Update security measures
   - Post-incident review

## Security Resources

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CWE/SANS Top 25: https://cwe.mitre.org/top25/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- Security Best Practices: https://cheatsheetseries.owasp.org/

---

**Last Updated**: January 15, 2024
**Maintained By**: Security Team
