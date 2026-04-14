# Database Operations Guide

## Database Schema

### Current Schema (v1.0)

#### Table: users
```sql
CREATE TABLE users (
  id INT PRIMARY KEY AUTO_INCREMENT,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  role ENUM('admin', 'manager', 'viewer') DEFAULT 'viewer',
  is_active BOOLEAN DEFAULT TRUE,
  last_login TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_is_active ON users(is_active);
```

#### Table: cameras
```sql
CREATE TABLE cameras (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  location VARCHAR(255),
  rtsp_url VARCHAR(500),
  status ENUM('active', 'inactive', 'error') DEFAULT 'inactive',
  is_active BOOLEAN DEFAULT TRUE,
  last_heartbeat TIMESTAMP,
  created_by INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (created_by) REFERENCES users(id)
);

CREATE INDEX idx_is_active ON cameras(is_active);
CREATE INDEX idx_status ON cameras(status);
```

#### Table: workers
```sql
CREATE TABLE workers (
  id INT PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(255) NOT NULL,
  employee_id VARCHAR(100) UNIQUE,
  email VARCHAR(255),
  phone VARCHAR(20),
  department VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX idx_employee_id ON workers(employee_id);
CREATE INDEX idx_email ON workers(email);
```

#### Table: violations
```sql
CREATE TABLE violations (
  id INT PRIMARY KEY AUTO_INCREMENT,
  violation_type ENUM(
    'NO_HARD_HAT',
    'NO_SAFETY_VEST',
    'NO_SAFETY_SHOES',
    'UNSAFE_POSTURE',
    'BLOCKED_EXIT',
    'FIRE_HAZARD',
    'OTHER'
  ) NOT NULL,
  severity ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') DEFAULT 'MEDIUM',
  status ENUM('open', 'investigating', 'resolved') DEFAULT 'open',
  camera_id INT NOT NULL,
  worker_id INT,
  confidence FLOAT,
  image_path VARCHAR(500),
  description TEXT,
  resolution_notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE,
  FOREIGN KEY (worker_id) REFERENCES workers(id)
);

CREATE INDEX idx_camera_date ON violations(camera_id, created_at);
CREATE INDEX idx_status ON violations(status);
CREATE INDEX idx_severity ON violations(severity);
CREATE INDEX idx_violation_type ON violations(violation_type);
```

#### Table: alerts
```sql
CREATE TABLE alerts (
  id INT PRIMARY KEY AUTO_INCREMENT,
  violation_id INT NOT NULL,
  severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
  status ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
  assigned_to INT,
  message TEXT,
  is_read BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  acknowledged_at TIMESTAMP,
  FOREIGN KEY (violation_id) REFERENCES violations(id) ON DELETE CASCADE,
  FOREIGN KEY (assigned_to) REFERENCES users(id)
);

CREATE INDEX idx_severity ON alerts(severity);
CREATE INDEX idx_status ON alerts(status);
CREATE INDEX idx_is_read ON alerts(is_read);
CREATE INDEX idx_created_at ON alerts(created_at);
```

#### Table: incidents
```sql
CREATE TABLE incidents (
  id INT PRIMARY KEY AUTO_INCREMENT,
  violations_count INT DEFAULT 0,
  severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
  status ENUM('open', 'investigating', 'resolved') DEFAULT 'open',
  location VARCHAR(255),
  description TEXT,
  reporter_id INT,
  assigned_to INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  resolved_at TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (reporter_id) REFERENCES users(id),
  FOREIGN KEY (assigned_to) REFERENCES users(id)
);

CREATE INDEX idx_status ON incidents(status);
CREATE INDEX idx_severity ON incidents(severity);
CREATE INDEX idx_created_at ON incidents(created_at);
```

## Backup & Recovery

### Automated Daily Backup

Create backup script: `scripts/backup.sh`

```bash
#!/bin/bash

BACKUP_DIR="/backups/database"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_NAME="safety_ai"
DB_USER="safety_user"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
mysqldump \
  -u $DB_USER \
  -p$DB_PASSWORD \
  --all-databases \
  --single-transaction \
  --quick \
  --lock-tables=false | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Backup files
tar -czf $BACKUP_DIR/files_$TIMESTAMP.tar.gz data/

# Keep only last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

# Verify backup
BACKUP_SIZE=$(ls -lh $BACKUP_DIR/backup_$TIMESTAMP.sql.gz | awk '{print $5}')
echo "Backup completed: $BACKUP_SIZE"

# Upload to cloud (optional)
# aws s3 cp $BACKUP_DIR/backup_$TIMESTAMP.sql.gz s3://my-bucket/backups/
```

### Schedule Backup

**Linux/Mac:**
Add to crontab:
```bash
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

**Windows:**
Create scheduled task:
```powershell
# PowerShell as Admin
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "C:\scripts\backup.ps1"
Register-ScheduledTask -TaskName "DBBackup" -Trigger $trigger -Action $action
```

### Restore from Backup

```bash
# Decompress backup
gunzip -c backup_20240115_020000.sql.gz > backup.sql

# Restore database
mysql -u root -p < backup.sql

# Or restore to specific database
mysql -u root -p safety_ai < backup.sql
```

## Migration Guide

### Migration v1.0 -> v1.1 (Future)

When new schema changes are needed:

```bash
# Create migration script
cat > migrations/001_add_audit_log.sql << EOF
-- Create audit log table
CREATE TABLE audit_log (
  id INT PRIMARY KEY AUTO_INCREMENT,
  table_name VARCHAR(255),
  action ENUM('INSERT', 'UPDATE', 'DELETE'),
  record_id INT,
  changes JSON,
  user_id INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add audit trigger
CREATE TRIGGER violations_audit_insert AFTER INSERT ON violations
FOR EACH ROW
BEGIN
  INSERT INTO audit_log (table_name, action, record_id, changes)
  VALUES ('violations', 'INSERT', NEW.id, JSON_OBJECT('new', ROW_TO_JSON(NEW)));
END;
EOF

# Test migration (backup first!)
mysql -u root -p safety_ai < migrations/001_add_audit_log.sql

# Verify
SELECT * FROM audit_log LIMIT 1;
```

## Performance Optimization

### Analyze Query Performance

```sql
-- Enable query logging
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 2;

-- Check slow log
SHOW VARIABLES LIKE 'slow_query_log%';

-- View slow queries
SELECT * FROM mysql.slow_log LIMIT 10;
```

### Add Indexes

```sql
-- Current indexes
SHOW INDEX FROM violations;
SHOW INDEX FROM cameras;

-- Add useful indexes
ALTER TABLE violations ADD INDEX idx_camera_date (camera_id, created_at);
ALTER TABLE violations ADD INDEX idx_status_resolved (status, resolved_at);
ALTER TABLE alerts ADD INDEX idx_severity_date (severity, created_at);
ALTER TABLE cameras ADD INDEX idx_status (status);

-- Check index usage
SELECT OBJECT_SCHEMA, OBJECT_NAME, COUNT_STAR
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE OBJECT_SCHEMA = 'safety_ai'
ORDER BY COUNT_STAR DESC;
```

### Optimize Storage

```sql
-- Check table sizes
SELECT TABLE_NAME, ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) as size_mb
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'safety_ai'
ORDER BY size_mb DESC;

-- Optimize tables
OPTIMIZE TABLE violations;
OPTIMIZE TABLE alerts;
OPTIMIZE TABLE cameras;
```

## Maintenance Tasks

### Daily Maintenance

```bash
#!/bin/bash

# Check database status
mysql -u root -p -e "SHOW STATUS LIKE 'Threads%';"

# Check connections
mysql -u root -p -e "SHOW PROCESSLIST;"

# Kill long-running queries (>5 minutes)
mysql -u root -p -e "SELECT ID FROM INFORMATION_SCHEMA.PROCESSLIST 
WHERE TIME > 300 AND COMMAND NOT IN ('Sleep') \G" | awk '/ID/{print $2}' | \
while read id; do
  mysql -u root -p -e "KILL $id;"
done
```

### Weekly Maintenance

```bash
#!/bin/bash

# Analyze all tables
mysql -u root -p safety_ai -e "
  SELECT CONCAT('ANALYZE TABLE ', TABLE_NAME, ';')
  FROM INFORMATION_SCHEMA.TABLES
  WHERE TABLE_SCHEMA = 'safety_ai'
" | grep ANALYZE | mysql -u root -p safety_ai

# Check for fragmentation
mysql -u root -p -e "
  SELECT TABLE_NAME, ROUND(((DATA_FREE / 1024 / 1024)), 2) as fragmented_mb
  FROM INFORMATION_SCHEMA.TABLES
  WHERE DATA_FREE > 0 AND TABLE_SCHEMA = 'safety_ai'
  ORDER BY fragmented_mb DESC;
"
```

### Monthly Maintenance

```bash
#!/bin/bash

# Full backup with binary log
mysqldump \
  -u root -p \
  --single-transaction \
  --all-databases \
  --flush-logs \
  --master-data=2 | gzip > full_backup_$(date +%Y%m%d).sql.gz

# Verify backup integrity
gunzip -t full_backup_$(date +%Y%m%d).sql.gz

# Archive old logs
find logs/ -mtime +90 -exec gzip {} \;

# Document schema
mysqldump -u root -p --no-data safety_ai > schema_backup_$(date +%Y%m%d).sql
```

## Scaling Strategies

### Vertical Scaling

Increase server resources:
```bash
# Current resources
free -h
df -h

# Recommended scaling
# < 1000 violations/day: 2GB RAM, 20GB disk
# < 10000 violations/day: 4GB RAM, 50GB disk
# > 10000 violations/day: 8GB+ RAM, 100GB+ disk
```

### Horizontal Scaling

Add read replicas:

```bash
# Master configuration
[mysqld]
server-id = 1
log-bin = /var/log/mysql/mysql-bin.log

# Slave configuration
# server-id = 2
# relay-log = /var/log/mysql/mysql-relay-bin
# read-only = ON

# Setup replication
CHANGE MASTER TO
  MASTER_HOST='master-ip',
  MASTER_USER='replication_user',
  MASTER_PASSWORD='password',
  MASTER_LOG_FILE='mysql-bin.000001',
  MASTER_LOG_POS=0;

START SLAVE;
SHOW SLAVE STATUS;
```

## Monitoring

### Key Metrics

```sql
-- Monitor connections
SHOW PROCESSLIST;
SHOW STATUS LIKE 'Threads%';

-- Monitor queries
SHOW STATUS LIKE 'Questions';
SHOW STATUS LIKE 'Slow_queries';

-- Monitor cache
SHOW STATUS LIKE 'Qcache%';

-- Monitor replication lag
SHOW SLAVE STATUS\G
```

### Alert Thresholds

- Connection limit: > 80% of max_connections
- Slave lag: > 5 seconds
- Slow queries: > 10 per minute
- Disk usage: > 85%
- Query time: > 2 seconds

### Monitoring Script

```bash
#!/bin/bash

THRESHOLD_LOAD=4
THRESHOLD_DISK=85

# Check load average
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}')
if (( $(echo "$LOAD > $THRESHOLD_LOAD" | bc -l) )); then
  echo "HIGH LOAD: $LOAD" | mail -s "Alert" admin@example.com
fi

# Check disk usage
DISK=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK -gt $THRESHOLD_DISK ]; then
  echo "HIGH DISK USAGE: $DISK%" | mail -s "Alert" admin@example.com
fi

# Check MySQL
mysql -u root -p -e "SHOW STATUS;" | grep Threads_connected | while read line; do
  THREADS=$(echo $line | awk '{print $2}')
  if [ $THREADS -gt 50 ]; then
    echo "HIGH THREADS: $THREADS" | mail -s "Alert" admin@example.com
  fi
done
```

## Emergency Procedures

### Database Corruption

```sql
-- Check for corruption
CHECK TABLE violations;
CHECK TABLE cameras;
CHECK TABLE workers;

-- Repair table
REPAIR TABLE violations;

-- If repair fails, restore from backup
```

### Connection Pool Exhausted

```sql
-- Find idle connections
SHOW PROCESSLIST;

-- Kill idle connections
KILL CONNECTION_ID;

-- Adjust pool settings
SET GLOBAL max_connections = 200;
SET GLOBAL max_allowed_packet = 256M;
```

### Disk Space Critical

```bash
# Check largest tables
SELECT TABLE_NAME, ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024 / 1024), 2) as size_gb
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'safety_ai'
ORDER BY size_gb DESC;

# Archive old data
DELETE FROM violations WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
DELETE FROM alerts WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);

# Optimize
OPTIMIZE TABLE violations;
OPTIMIZE TABLE alerts;
```

## Backup Verification Checklist

- [x] Backup file created
- [x] File size reasonable (> 1MB)
- [x] Compression successful
- [x] Can decompress without errors
- [x] Can restore successfully
- [x] Data integrity verified
- [x] All tables present
- [x] All records count correct
- [x] File uploaded to cloud
- [x] Retention policy applied

---

**Last Updated**: January 15, 2024
**Maintained By**: Database Team
