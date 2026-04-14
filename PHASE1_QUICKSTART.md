# PHASE 1: QUICK START - Right Now

## Your Action Items (Next 30 minutes)

### ✅ STEP 1: Verify System is Running

**Run these commands in separate terminal windows:**

```powershell
# Terminal 1: Check Backend Health
curl http://localhost:8000/health

# Expected: Shows status, database connection, version
# If ERROR: Backend not running → Run: cd backend && python server.py
```

```powershell
# Terminal 2: Check Frontend
curl http://localhost:3000

# Expected: Returns HTML
# If ERROR: Frontend not running → Run: cd frontend && npm start
```

```powershell
# Terminal 3: Check Database
mysql -u safety_user -p safety_ai -e "SELECT 'Database Connected!' as Status;"

# Enter password: [from .env file]
# Expected: Shows "Database Connected!"
# If ERROR: MySQL not running → Run: sudo systemctl start mysql
```

---

### ✅ STEP 2: Document Current Status

**Create a file: `CURRENT_STATUS.txt` in project root**

```
SYSTEM AUDIT - April 14, 2026

BACKEND STATUS:
- URL: http://localhost:8000
- Health Check: [PASS/FAIL]
- Port: [8000]
- Database Connected: [YES/NO]

FRONTEND STATUS:
- URL: http://localhost:3000
- Loads: [YES/NO]
- Port: [3000]

DATABASE STATUS:
- Status: [CONNECTED/ERROR]
- User: safety_user
- Database: safety_ai
- Tables: [COUNT: ?]

CURRENT HOSTNAME:
[your-domain-or-localhost]

CURRENT TIME:
[System time]
```

---

### ✅ STEP 3: Generate Production Credentials

**Run these Python commands to generate strong keys:**

```powershell
# Terminal: Generate Keys
python

# In Python interactive mode, run:
import secrets

# Generate SECRET_KEY
secret_key = secrets.token_urlsafe(32)
print("SECRET_KEY=" + secret_key)

# Generate strong database password
db_password = secrets.token_urlsafe(16)
print("DB_PASSWORD=" + db_password)

# Exit Python
exit()
```

**SAVE these values! You'll need them for .env**

---

### ✅ STEP 4: Prepare .env for Production

**Create a new file: `.env.production`**

```bash
# Copy from existing .env but change these values:

# SECURITY - Use values from Step 3
ENV=production
DEBUG=false
SECRET_KEY=<PASTE-FROM-STEP-3-SECRET_KEY>

# DATABASE - Use values from Step 3
DB_PASSWORD=<PASTE-FROM-STEP-3-DB_PASSWORD>

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS - Change to your domain
CORS_ORIGINS=["https://yourdomain.com"]

# JWT
JWT_EXPIRE_MINUTES=1440

# RATE LIMITING
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# LOGGING
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_FORMAT=json
```

**IMPORTANT:** Keep `.env.production` secure! Don't commit to Git!

---

### ✅ STEP 5: Quick Code Quality Check

**Clean up code (5 minutes):**

```powershell
# Backend cleanup
cd backend
pip install black flake8
black .
flake8 . --max-line-length=100

# Frontend cleanup
cd ../frontend
npx prettier --write src/
```

---

### ✅ STEP 6: Run Quick Tests

**Test critical paths (10 minutes):**

```powershell
# Backend API test
curl -X GET http://localhost:8000/api/violations

# Frontend load test
curl -s http://localhost:3000 | head -20

# Database connection test
mysql -u safety_user -p safety_ai -e "SHOW TABLES;"
```

---

## ✨ Phase 1 Complete!

When you've done steps 1-6, you're ready for **Phase 2: Security Hardening**.

---

# NOW: PHASE 1 COMPLETE CHECKLIST

Can you confirm these are done? Reply with YES/NO for each:

1. ✅ Backend health check working: **[YES/NO]**
2. ✅ Frontend page loading: **[YES/NO]**
3. ✅ Database connected: **[YES/NO]**
4. ✅ Production credentials generated: **[YES/NO]**
5. ✅ .env.production created: **[YES/NO]**
6. ✅ Code cleanup done: **[YES/NO]**
7. ✅ Quick tests passed: **[YES/NO]**

**Once ALL are YES, I'll guide you through PHASE 2: Security Hardening**

---

## If ANY step failed, here's the fix:

### "Backend health check ERROR"
```powershell
cd backend
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
python server.py
```

### "Frontend page not loading"
```powershell
cd frontend
npm install
npm start
```

### "Database connection ERROR"
```powershell
# Windows command to start MySQL
net start MySQL80

# Or in Linux/Mac
sudo systemctl start mysql
```

### "Python import error"
```powershell
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

**Ready to start? Reply with your Phase 1 checklist results!**
