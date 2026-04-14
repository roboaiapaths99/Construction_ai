# Frontend Production Guide

## Overview

This guide covers production-ready best practices for the React frontend.

## Pre-Production Checklist

### Code Quality
- [ ] All console.log statements removed or converted to proper logging
- [ ] No hardcoded API URLs (use environment variables)
- [ ] All error cases handled
- [ ] Loading states implemented
- [ ] All images optimized
- [ ] Code linting passing (ESLint)
- [ ] No unused dependencies

### Performance
- [ ] Minification enabled
- [ ] Tree-shaking configured
- [ ] Code splitting implemented
- [ ] Images optimized (WebP format)
- [ ] Lazy loading enabled for routes
- [ ] CSS minified
- [ ] Bundle size < 500KB (gzipped)

### Security
- [ ] Remove sensitive data from source
- [ ] Enable Content Security Policy
- [ ] Implement helmet.js
- [ ] Validate all user inputs
- [ ] Escape all user-generated content
- [ ] No inline scripts
- [ ] HTTPS enforced
- [ ] API keys in environment variables only

### Accessibility
- [ ] WCAG 2.1 AA compliance
- [ ] Keyboard navigation works
- [ ] Screen reader compatible
- [ ] Color contrast adequate
- [ ] Mobile responsive
- [ ] Touch targets 44x44 minimum

### Testing
- [ ] Unit tests coverage > 80%
- [ ] Integration tests pass
- [ ] E2E tests automated
- [ ] Cross-browser tested

## Building for Production

### 1. Environment Configuration

Create `.env.production`:
```bash
REACT_APP_API_URL=https://api.yourdomain.com/api
REACT_APP_ENV=production
REACT_APP_LOG_LEVEL=error
REACT_APP_ENABLE_SENTRY=true
REACT_APP_SENTRY_DSN=<your-sentry-dsn>
```

### 2. Build Optimization

Update `package.json`:
```json
{
  "scripts": {
    "build": "react-scripts build",
    "build:analyze": "source-map-explorer 'build/static/js/*.js'",
    "build:production": "GENERATE_SOURCEMAP=false npm run build"
  }
}
```

### 3. Build Command
```bash
npm run build:production
```

This creates an optimized `build/` directory ready for deployment.

## Deployment Options

### Option 1: Nginx (Recommended)

#### 1. Nginx Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name app.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/your-cert.crt;
    ssl_certificate_key /etc/ssl/private/your-key.key;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.segment.com; style-src 'self' 'unsafe-inline';" always;
    
    # Gzip compression
    gzip on;
    gzip_types text/plain text/css text/javascript application/json application/javascript;
    gzip_min_length 1000;
    gzip_comp_level 6;
    
    # Cache static assets
    location ~* \.(jpg|jpeg|png|gif|svg|webp|ico|css|js|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "immutable";
    }
    
    # Serve React app
    root /var/www/app;
    location / {
        try_files $uri $uri/ /index.html;
        expires -1;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    # Proxy API requests
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 2. Deploy
```bash
npm run build:production
sudo cp -r build/* /var/www/app/
sudo systemctl restart nginx
```

### Option 2: Docker

#### 1. Create Dockerfile
```dockerfile
# Build stage
FROM node:18-alpine as builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ENV REACT_APP_API_URL=/api
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 2. Build and Run
```bash
docker build -t safety-frontend:latest .
docker run -p 80:80 safety-frontend:latest
```

### Option 3: Vercel/Netlify

#### Deploy to Vercel
```bash
npm install -g vercel
vercel --prod
```

#### Deploy to Netlify
```bash
npm run build
netlify deploy --prod --dir=build
```

## Performance Optimization

### 1. Code Splitting

Implement for large pages:
```javascript
import React, { Suspense } from 'react';
import { lazy } from 'react';

const Violations = lazy(() => import('./pages/Violations'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Violations />
    </Suspense>
  );
}
```

### 2. Image Optimization

Convert images to WebP:
```bash
# Install imagemin
npm install imagemin imagemin-webp

# Convert images
npx imagemin src/assets/images/*.png --out-dir=src/assets/images --plugin=webp
```

Use in components:
```javascript
<picture>
  <source srcSet="image.webp" type="image/webp" />
  <source srcSet="image.png" type="image/png" />
  <img src="image.png" alt="Description" />
</picture>
```

### 3. Bundle Analysis

```bash
npm run build:analyze
```

### 4. Performance Monitoring

Add to `public/index.html`:
```html
<script>
  // Web Vitals measurement
  window.addEventListener('load', () => {
    const perfData = performance.timing;
    const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
    console.log('Page Load Time: ' + pageLoadTime);
    
    // Send to analytics
    if (window.gtag) {
      gtag('event', 'page_view', {
        'page_load_time': pageLoadTime
      });
    }
  });
</script>
```

## Monitoring & Analytics

### 1. Sentry (Error Tracking)

Install:
```bash
npm install @sentry/react
```

Configure:
```javascript
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: process.env.REACT_APP_SENTRY_DSN,
  environment: process.env.REACT_APP_ENV,
  tracesSampleRate: 1.0,
});
```

### 2. Google Analytics

Add to `public/index.html`:
```html
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_ID');
</script>
```

### 3. Application Insights (Azure)

```bash
npm install appinsights
```

## Security Best Practices

### 1. Content Security Policy

Add to `public/index.html`:
```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-inline' cdn.segment.com; 
               style-src 'self' 'unsafe-inline'; 
               img-src 'self' https:; 
               font-src 'self' https://fonts.googleapis.com;">
```

### 2. Environment Variables

Never commit sensitive data. Use `.env.local` (gitignored):
```bash
# .env.production
REACT_APP_API_URL=https://api.yourdomain.com/api

# .env.local (not in git)
REACT_APP_SECRET_KEY=...
```

### 3. Input Validation

Always validate user input:
```javascript
const validateEmail = (email) => {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
};

const handleSubmit = (e) => {
  if (!validateEmail(email)) {
    toast.error('Invalid email');
    return;
  }
  // Submit
};
```

## Logging & Debugging

### 1. Structured Logging

```javascript
// logger.js
const log = (level, message, data = {}) => {
  const logEntry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    data,
    url: window.location.href
  };
  
  if (process.env.REACT_APP_ENV === 'production') {
    // Send to logging service
    fetch('/api/logs', { method: 'POST', body: JSON.stringify(logEntry) });
  } else {
    console.log(logEntry);
  }
};
```

### 2. Error Boundary

Already implemented - captures unhandled errors

## Maintenance

### Weekly
- Check error logs
- Review performance metrics
- Test critical user flows

### Monthly
- Update dependencies
- Security audit
- Performance optimization review

### Quarterly
- Full regression testing
- UX optimization review
- Accessibility audit

## Support

For frontend issues, contact: frontend-support@constructionsafety.ai

## Additional Resources

- React Documentation: https://react.dev/
- Nginx Documentation: https://nginx.org/en/docs/
- Web Vitals: https://web.dev/vitals/
- Security Headers: https://securityheaders.com/
