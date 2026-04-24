# Construction AI Frontend

A modern, production-ready React frontend for the Construction AI Safety Monitoring System.

## Features

### 🏗️ Real-time Dashboard
- Live statistics with auto-refresh
- Interactive charts and visualizations
- System status monitoring
- Recent violations and alerts overview

### ⚠️ Violations Management
- Comprehensive violation tracking
- Status management (Open, Resolved, Investigating)
- Evidence viewing with image support
- Advanced filtering and search
- Real-time status updates

### 👷 Workers Monitoring
- Worker status tracking (Active, At Risk)
- Role-based categorization
- Location and group size information
- Real-time status updates

### 🚨 Alert System
- Priority-based alert management
- Sound notifications for high-priority alerts
- Alert dismissal and clearing
- Real-time alert polling

### 📹 Camera Management
- Camera status monitoring (Online/Offline)
- Recording control (Start/Stop)
- Live view integration ready
- Camera settings management

### ⚙️ System Settings
- Notification preferences
- Dashboard configuration
- Camera settings
- System preferences

## Technology Stack

- **React 18** - Modern UI framework
- **React Router** - Client-side routing
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Icon library
- **Axios** - HTTP client
- **Custom Hooks** - API management and polling

## Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn
- Backend API running on http://localhost:8000

### Installation

1. Clone the repository
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Create environment file:
   ```bash
   cp .env.example .env
   ```

5. Start the development server:
   ```bash
   npm start
   ```

The application will be available at http://localhost:3000

## Environment Variables

Create a `.env` file in the root directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000/ws
REACT_APP_REFRESH_INTERVAL=5000
```

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm run eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Layout.jsx      # Main application layout
│   ├── StatCard.jsx    # Statistics card component
│   ├── LoadingSpinner.jsx
│   └── AlertBanner.jsx
├── pages/              # Page components
│   ├── Dashboard.jsx   # Main dashboard
│   ├── Violations.jsx  # Violations management
│   ├── Workers.jsx     # Worker monitoring
│   ├── Alerts.jsx      # Alert system
│   ├── Cameras.jsx     # Camera management
│   └── Settings.jsx    # System settings
├── hooks/              # Custom React hooks
│   └── useApi.js       # API and polling hooks
├── api/                # API configuration
│   └── index.js        # API endpoints and configuration
├── App.jsx             # Main App component
├── index.js            # Application entry point
└── index.css           # Global styles and Tailwind
```

## API Integration

The frontend is designed to work with the FastAPI backend. Key API endpoints:

- `/health` - Health check
- `/incidents` - Incident management
- `/violations` - Violations data
- `/workers` - Worker data
- `/alerts` - Alert data
- `/cameras` - Camera data
- `/dashboard/stats` - Dashboard statistics

## Real-time Features

- **Auto-refresh**: Dashboard and data pages automatically refresh
- **Polling**: Configurable polling intervals for real-time updates
- **Status Updates**: Real-time status updates for violations and workers
- **Alert Notifications**: Sound and visual notifications for high-priority alerts

## Responsive Design

- Mobile-first approach
- Responsive grid layouts
- Touch-friendly interface
- Adaptive navigation

## Production Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized build in the `build` directory.

### Deployment Options

1. **Static Hosting** (Recommended)
   - Deploy to Netlify, Vercel, or similar
   - Upload the `build` directory
   - Configure API endpoint in environment

2. **Docker Deployment**
   ```dockerfile
   FROM node:16-alpine as build
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   COPY . .
   RUN npm run build
   
   FROM nginx:alpine
   COPY --from=build /app/build /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   EXPOSE 80
   CMD ["nginx", "-g", "daemon off;"]
   ```

3. **Server-side Rendering**
   - Can be adapted for Next.js if needed
   - Current version is client-side rendered

## Performance Optimizations

- Code splitting by route
- Lazy loading of components
- Optimized bundle size
- Efficient polling intervals
- Memoized components

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Security Features

- CORS configuration
- Input validation
- XSS protection
- Secure API communication

## Contributing

1. Follow React best practices
2. Use TypeScript for new components
3. Maintain responsive design
4. Test on multiple screen sizes
5. Update documentation

## License

This project is part of the Construction AI Safety Monitoring System.
