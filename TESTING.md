# Testing Strategy & Guide

## Overview

This document provides comprehensive testing strategies for the AI Construction Safety System across all layers.

## Testing Pyramid

```
        ┌─────────────────┐
        │   E2E Tests     │  10% - Full workflow testing
        ├─────────────────┤
        │ Integration     │  30% - API & database testing
        │ Tests           │
        ├─────────────────┤
        │ Unit Tests      │  60% - Individual function testing
        └─────────────────┘
```

## Unit Tests

### Backend Unit Tests

Location: `backend/tests/`

```python
# Example: Test authentication
import pytest
from config.auth import AuthManager

def test_password_hashing():
    auth = AuthManager()
    password = "test123!@#"
    hash = auth.hash_password(password)
    assert auth.verify_password(password, hash)
    assert not auth.verify_password("wrong", hash)

def test_jwt_token_creation():
    auth = AuthManager()
    token = auth.create_access_token({"user_id": 1})
    payload = auth.verify_token(token)
    assert payload["user_id"] == 1

def test_token_expiration():
    auth = AuthManager()
    expired_token = auth.create_access_token({"user_id": 1}, expire_minutes=0)
    with pytest.raises(Exception):
        auth.verify_token(expired_token)
```

### Frontend Unit Tests

Location: `frontend/src/__tests__/`

```javascript
// Example: Test API utility
import { validateEmail, parseApiError } from '../utils/validators';

describe('Email Validation', () => {
  test('validates correct email', () => {
    expect(validateEmail('user@example.com')).toBe(true);
  });

  test('rejects invalid email', () => {
    expect(validateEmail('invalid.email')).toBe(false);
  });

  test('handles edge cases', () => {
    expect(validateEmail('')).toBe(false);
    expect(validateEmail(null)).toBe(false);
  });
});
```

### Run Unit Tests

**Backend:**
```bash
cd backend
pytest -v
```

**Frontend:**
```bash
cd frontend
npm test -- --coverage
```

## Integration Tests

### API Integration Tests

Location: `backend/tests/integration/`

```python
import pytest
from fastapi.testclient import TestClient
from server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_get_violations_list(client):
    response = client.get("/api/violations")
    assert response.status_code == 200
    assert "data" in response.json()
    assert "total" in response.json()

def test_create_incident(client):
    incident_data = {
        "violation_type": "NO_HARD_HAT",
        "severity": "HIGH",
        "camera_id": 1,
        "confidence": 0.95
    }
    response = client.post("/api/incidents", json=incident_data)
    assert response.status_code == 201
    assert response.json()["id"] > 0
```

### Database Integration Tests

```python
def test_database_connection(db):
    """Test database connectivity"""
    result = db.execute("SELECT 1")
    assert result.fetchone() is not None

def test_violation_cascade_delete():
    """Test cascade delete behavior"""
    # Create and delete camera
    # Verify related violations deleted
    pass
```

### API Authentication Tests

```python
def test_api_requires_authentication(client):
    """Test protected endpoints require auth"""
    response = client.get("/api/violations")
    assert response.status_code == 401

def test_invalid_token_rejected(client):
    response = client.get(
        "/api/violations",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
```

## End-to-End Tests

### User Flow Tests

Location: `frontend/e2e/`

```javascript
// Example: Complete workflow
describe('Complete Violation Workflow', () => {
  test('user can view and respond to violations', async () => {
    // 1. Login
    await page.goto('http://localhost:3000/login');
    await page.type('[name="email"]', 'admin@example.com');
    await page.type('[name="password"]', 'password123');
    await page.click('[type="submit"]');
    
    // 2. Wait for dashboard
    await page.waitForSelector('[data-testid="dashboard"]');
    
    // 3. Navigate to violations
    await page.click('a[href="/violations"]');
    
    // 4. Verify violations loaded
    const violations = await page.$$('.violation-item');
    expect(violations.length).toBeGreaterThan(0);
    
    // 5. Click first violation
    await violations[0].click();
    
    // 6. Update status
    await page.select('[name="status"]', 'RESOLVED');
    await page.click('[type="submit"]');
    
    // 7. Verify success
    await page.waitForSelector('.success-message');
  });
});
```

### Run E2E Tests

```bash
cd frontend
npm run e2e
```

## Performance Tests

### Load Testing

Using Apache Bench:

```bash
# Single request
ab -n 1 http://localhost:8000/api/violations

# 100 requests, 10 concurrent
ab -n 100 -c 10 http://localhost:8000/api/violations

# 1000 requests, 50 concurrent
ab -n 1000 -c 50 http://localhost:8000/api/violations
```

### Expected Results

| Metric | Target | Acceptable |
|--------|--------|-----------|
| Response Time (p50) | <100ms | <200ms |
| Response Time (p95) | <200ms | <500ms |
| Response Time (p99) | <300ms | <1000ms |
| Error Rate | <0.1% | <1% |
| Throughput | 100+ req/s | 50+ req/s |

### Database Query Performance

```sql
-- Find slow queries
SELECT * FROM mysql.slow_log WHERE query_time > 1;

-- Index analysis
EXPLAIN SELECT * FROM violations WHERE camera_id = 1 AND created_at > NOW() - INTERVAL 7 DAY;

-- Check index usage
ANALYZE TABLE violations;
```

## Security Testing

### OWASP Top 10 Checks

```python
# 1. SQL Injection
def test_sql_injection_prevention():
    response = client.get("/api/violations?camera_id=1 OR 1=1")
    assert response.status_code == 200
    # Should be safe, no error
    
# 2. XSS Prevention
def test_xss_prevention():
    data = {"description": "<script>alert('xss')</script>"}
    response = client.post("/api/incidents", json=data)
    # Should sanitize or reject
    
# 3. Authentication Bypass
def test_auth_bypass_prevention():
    response = client.get(
        "/api/admin",
        headers={"User-Agent": "curl"}  # No auth token
    )
    assert response.status_code == 401
```

### Password Security Tests

```python
def test_password_requirements():
    """Test password meets security requirements"""
    weak_passwords = [
        "weak",
        "123456",
        "password",
        "qwerty"
    ]
    for pwd in weak_passwords:
        assert not meets_password_requirements(pwd)
    
    strong_password = "MyP@ssw0rd123!"
    assert meets_password_requirements(strong_password)
```

### Rate Limiting Tests

```python
def test_rate_limiting(client):
    """Test rate limit enforcement"""
    for i in range(101):  # Over limit
        response = client.get("/api/violations")
    # 101st request should be rate limited
    assert response.status_code == 429
```

## Behavior-Driven Testing

### BDD Scenarios

Location: `backend/tests/features/`

```gherkin
Feature: Violation Detection
  Scenario: Detect safety violation
    Given a camera is monitoring
    When a violation occurs
    Then an alert is created
    And notification is sent
    And violation is logged

  Scenario: Filter violations by type
    Given 10 violations exist
    When filtering by type NO_HARD_HAT
    Then only matching violations returned
```

## Test Data Management

### Fixtures

```python
@pytest.fixture
def test_camera(db):
    """Create test camera"""
    camera = Camera(
        name="Test Camera",
        location="Test Site",
        rtsp_url="rtsp://localhost/test",
        is_active=True
    )
    db.add(camera)
    db.commit()
    return camera

@pytest.fixture
def test_violations(db, test_camera):
    """Create test violations"""
    violations = [
        Violation(
            camera_id=test_camera.id,
            violation_type="NO_HARD_HAT",
            severity="HIGH"
        )
        for _ in range(5)
    ]
    db.add_all(violations)
    db.commit()
    return violations
```

### Cleanup

```python
@pytest.fixture(autouse=True)
def cleanup_db(db):
    """Clean database after each test"""
    yield
    db.query(Violation).delete()
    db.query(Camera).delete()
    db.commit()
```

## Continuous Testing

### CI/CD Pipeline

GitHub Actions workflow: `.github/workflows/tests.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mysql:
        image: mysql:8.0
        env:
          MYSQL_ROOT_PASSWORD: password
        options: >-
          --health-cmd="mysqladmin ping"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=3
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov
      
      - name: Run backend tests
        run: cd backend && pytest --cov
      
      - name: Set up Node
        uses: actions/setup-node@v2
        with:
          node-version: '18'
      
      - name: Install frontend dependencies
        run: cd frontend && npm install
      
      - name: Run frontend tests
        run: cd frontend && npm test -- --coverage
      
      - name: Run E2E tests
        run: cd frontend && npm run e2e
```

## Manual Testing Checklist

### Functionality Testing

- [ ] Create new violation record
- [ ] Update violation status
- [ ] Delete violation
- [ ] Search violations by date
- [ ] Filter by severity
- [ ] Export data to CSV
- [ ] View camera feed
- [ ] Create new camera
- [ ] Upload worker information
- [ ] Receive alert notifications
- [ ] View dashboard statistics
- [ ] Generate reports

### Usability Testing

- [ ] UI is intuitive
- [ ] Forms are easy to fill
- [ ] Error messages are clear
- [ ] Navigation is logical
- [ ] Page load times acceptable
- [ ] Mobile responsive
- [ ] Accessibility compliant

### Compatibility Testing

- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)
- [ ] Mobile Safari
- [ ] Mobile Chrome

## Test Coverage Goals

### Backend

- Unit tests: 80% coverage
- Integration tests: 95% coverage
- Critical paths: 100% coverage
- Authentication: 100% coverage

### Frontend

- Components: 75% coverage
- Utils: 85% coverage
- Hooks: 80% coverage
- Critical user flows: 100% coverage

### Check Coverage

**Backend:**
```bash
pytest --cov=backend --cov-report=html
```

**Frontend:**
```bash
npm test -- --coverage
```

## Test Execution Timeline

### Daily
- Unit tests (5 min)
- Critical integration tests (10 min)

### Twice Weekly
- Full unit test suite (15 min)
- Integration tests (20 min)
- E2E tests (30 min)

### Weekly
- Load testing
- Security scanning
- Performance profiling

### Monthly
- Full regression testing
- Manual testing checklist
- Accessibility audit
- Performance benchmarking

## Debugging Tests

### Backend Debug

```bash
# Run with verbose output
pytest -vv -s

# Run specific test
pytest backend/tests/test_auth.py::test_password_hashing -vv

# Run with pdb
pytest --pdb
```

### Frontend Debug

```bash
# Run with debug output
npm test -- --verbose

# Watch mode
npm test -- --watch

# Debug Chrome DevTools
node --inspect-brk node_modules/.bin/jest
```

## Known Issues & Workarounds

### Database Tests Timeout
- **Issue**: MySQL connection pool exhaustion
- **Workaround**: Increase pool size, use pytest-timeout

### Flaky E2E Tests
- **Issue**: Race conditions in async operations
- **Workaround**: Add explicit waits, use cy.intercept() for stubs

### Performance Test Variance
- **Issue**: Results vary by system load
- **Workaround**: Run tests in isolation, repeat 5 times for average

## Reporting

### Test Report Template

```
Test Run: 2024-01-15 14:30
Duration: 12 minutes
Environment: Ubuntu 22.04, Python 3.11, Node 18

Results:
- Unit Tests: 245 passed, 0 failed (98%)
- Integration Tests: 42 passed, 0 failed (100%)
- E2E Tests: 8 passed, 0 failed (100%)
- Coverage: 92%

Issues Found:
- None critical
- 2 UI improvements suggested

Performance:
- API p95 latency: 180ms ✅
- Database queries: avg 45ms ✅
- Frontend bundle: 340kb ✅
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Jest Documentation](https://jestjs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [React Testing Library](https://testing-library.com/react)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)

---

**Last Updated**: January 15, 2024
**Maintained By**: QA Team
