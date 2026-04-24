import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Violations from './pages/Violations';
import Workers from './pages/Workers';
import Attendance from './pages/Attendance';
import Alerts from './pages/Alerts';
import Cameras from './pages/Cameras';
import Settings from './pages/Settings';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './context/ToastContext';
import { AppProvider } from './context/AppContext';

function App() {
  return (
    <ErrorBoundary>
      <AppProvider>
        <ToastProvider>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/violations" element={<Violations />} />
                <Route path="/workers" element={<Workers />} />
                <Route path="/attendance" element={<Attendance />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/cameras" element={<Cameras />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Layout>
          </Router>
        </ToastProvider>
      </AppProvider>
    </ErrorBoundary>
  );
}

const NotFound = () => (
  <div className="flex items-center justify-center h-screen">
    <div className="text-center">
      <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>
      <p className="text-xl text-gray-600 mb-8">Page not found</p>
      <a href="/" className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700">
        Go back home
      </a>
    </div>
  </div>
);

export default App;
