# Construction AI Safety System - Frontend

A production-ready React application for real-time construction site safety monitoring with AI-powered detection capabilities.

## 🚀 Features

### Core Functionality
- ✅ **Real-time Dashboard** - Live monitoring of safety metrics
- ✅ **Violation Tracking** - Comprehensive violation management with status updates
- ✅ **Worker Monitoring** - Track worker safety status and compliance
- ✅ **Alert System** - Real-time alerts and notifications
- ✅ **Camera Management** - Monitor multiple camera feeds with live detection
- ✅ **Customizable Settings** - Full configuration management

### Production Features
- ✅ **Error Boundaries** - Graceful error handling
- ✅ **Toast Notifications** - User-friendly messaging system
- ✅ **Global State Management** - Context API for app-wide state
- ✅ **Pagination** - Efficient handling of large datasets
- ✅ **Data Export** - CSV, JSON, and Report generation
- ✅ **Responsive Design** - Mobile-friendly interface
- ✅ **Performance Optimized** - Memoization and code splitting
- ✅ **Accessibility** - WCAG compliant components
- ✅ **Environment Configuration** - Easy deployment configuration

## 📋 Prerequisites

- Node.js 14.x or higher
- npm 6.x or higher
- Backend API running on `http://localhost:8000`

## 🛠️ Installation

1. Clone the repository
```bash
cd frontend
```

2. Install dependencies
```bash
npm install
```

3. Create environment file
```bash
cp .env.example .env.local
```

4. Configure environment variables in `.env.local`
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=10000
REACT_APP_LOG_LEVEL=info
REACT_APP_DEBUG=false
REACT_APP_VERSION=1.0.0
```

## 🚀 Running the Application

### Development
```bash
npm start
```
Opens http://localhost:3000

### Production Build
```bash
npm run build
```
Creates optimized production build in `build` folder

### Testing
```bash
npm test
```

## 📁 Project Structure

```
src/
├── api/
│   └── index.js           # API configuration and endpoints
├── components/
│   ├── ErrorBoundary.jsx  # Error boundary component
│   ├── Layout.jsx         # Main layout wrapper
│   ├── Pagination.jsx     # Pagination component
│   ├── StatCard.jsx       # Statistics card
│   ├── LoadingSpinner.jsx # Loading indicator
│   ├── AlertBanner.jsx    # Alert messages
│   └── ...
├── context/
│   ├── ToastContext.jsx   # Toast notification system
│   └── AppContext.jsx     # Global app state
├── pages/
│   ├── Dashboard.jsx      # Main dashboard
│   ├── Violations.jsx     # Violations list & management
│   ├── Workers.jsx        # Worker monitoring
│   ├── Alerts.jsx         # Alert system
│   ├── Cameras.jsx        # Camera management
│   └── Settings.jsx       # Application settings
├── hooks/
│   ├── useApi.js          # API fetching hook
│   └── useApiStable.js    # Stable polling hook
├── utils/
│   └── exportUtils.js     # Export & report utilities
├── config/
│   └── index.js           # Configuration management
├── App.jsx                # Main app component
└── index.js               # Entry point
```

## 🎨 Key Components

### ErrorBoundary
Catches React errors and displays a fallback UI
```jsx
<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

### Toast Notifications
Display user-friendly messages
```jsx
const { success, error, warning, info } = useToast();
success('Operation completed!');
error('Something went wrong');
```

### Pagination
Handle large datasets efficiently
```jsx
<Pagination
  currentPage={page}
  totalPages={totalPages}
  onPageChange={setPage}
  totalItems={total}
/>
```

### Data Export
Export data to various formats
```jsx
import { exportToCSV, generateReport } from '../utils/exportUtils';
exportToCSV(data, 'filename');
generateReport('Title', sections);
```

### Global State
Access app-wide settings and preferences
```jsx
const { theme, autoRefresh, updateTheme } = useAppState();
```

## 🔌 API Endpoints

All endpoints are configured through the API module:

```javascript
api.health()                    // Health check
api.getDashboardStats()         // Dashboard statistics
api.getViolations()             // List of violations
api.getWorkers()                // Worker list
api.getAlerts()                 // Alert list
api.getCameras()                // Camera list
```

## 🎯 Performance Optimization

- **Code Splitting** - Routes are lazy-loaded
- **Memoization** - Heavy components are memoized
- **API Caching** - Smart polling intervals
- **Image Optimization** - Tailwind's image optimization
- **Bundle Analysis** - Use `npm run build -- --report` to analyze

## 🔐 Security

- Environment variables for sensitive data
- CORS enabled on backend
- Input validation on all forms
- XSS protection via React's built-in sanitization
- CSRF protection via API headers

## 🚀 Deployment

### Docker
```bash
docker build -t construction-ai-frontend .
docker run -p 3000:3000 construction-ai-frontend
```

### Vercel/Netlify
```bash
npm run build
# Deploy the build folder
```

### Traditional Server
```bash
npm run build
# Serve build folder with your web server (nginx, Apache, etc.)
```

## 📊 Configuration

Edit `src/config/index.js` for:
- API base URL
- API timeout
- Logging levels
- Polling intervals
- UI preferences

## 🐛 Troubleshooting

### API Connection Issues
- Check backend is running on configured URL
- Verify CORS headers on backend
- Check network tab in browser DevTools

### State Management Issues
- Ensure ToastProvider and AppProvider wrap your app
- Check localStorage for saved preferences
- Clear browser cache if settings don't load

### Performance Issues
- Check Network tab for large API responses
- Use React DevTools Profiler
- Check for unnecessary re-renders

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📝 License

Proprietary - Construction AI Safety System

## 📞 Support

For issues and support:
- Email: support@constructionai.com
- Documentation: /docs
- Issues: GitHub Issues

## 🗺️ Roadmap

- [ ] Dark mode theme
- [ ] PWA offline support
- [ ] Advanced analytics
- [ ] Multi-language support
- [ ] Real-time streaming optimization
- [ ] Mobile app
- [ ] Advanced reporting

## 📦 Dependencies

Key dependencies are managed in `package.json`:
- **react** - UI library
- **react-router-dom** - Routing
- **axios** - HTTP client
- **recharts** - Charts and graphs
- **lucide-react** - Icons
- **tailwindcss** - Styling

## 📄 Environment Variables

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=10000

# Logging
REACT_APP_LOG_LEVEL=info
REACT_APP_DEBUG=false

# App Info
REACT_APP_VERSION=1.0.0
```

---

**Version:** 1.0.0  
**Last Updated:** 2026-04-13
