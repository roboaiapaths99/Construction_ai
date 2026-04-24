import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import CameraFeed from './components/CameraFeed';
import DetectionResults from './components/DetectionResults';
import StatusPanel from './components/StatusPanel';

function App() {
  const [status, setStatus] = useState({ model_loaded: false, camera_running: false });
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);
  const statusCheckInterval = useRef(null);

  // Check backend status on mount
  useEffect(() => {
    checkBackendStatus();
    statusCheckInterval.current = setInterval(checkBackendStatus, 5000);
    
    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current);
      }
    };
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        if (data.last_detection && data.last_detection.detections) {
          setDetections(data.last_detection.detections);
        }
        setError(null);
      }
    } catch (err) {
      setError(`Backend not responding: ${err.message}`);
    }
  };

  const handleStartCamera = async () => {
    try {
      const response = await fetch('http://localhost:8000/camera/start', { method: 'GET' });
      if (response.ok) {
        await checkBackendStatus();
        setError(null);
      }
    } catch (err) {
      setError(`Failed to start camera: ${err.message}`);
    }
  };

  const handleStopCamera = async () => {
    try {
      const response = await fetch('http://localhost:8000/camera/stop', { method: 'GET' });
      if (response.ok) {
        await checkBackendStatus();
        setError(null);
      }
    } catch (err) {
      setError(`Failed to stop camera: ${err.message}`);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛡️ AI Safety Monitoring System v2.0</h1>
        <p>Simple, Reliable, Error-Free</p>
      </header>

      <main className="app-main">
        {error && <div className="error-banner">{error}</div>}

        <div className="container">
          <div className="left-panel">
            <StatusPanel 
              status={status} 
              onStartCamera={handleStartCamera}
              onStopCamera={handleStopCamera}
            />
          </div>

          <div className="middle-panel">
            <CameraFeed isRunning={status.camera_running} />
          </div>

          <div className="right-panel">
            <DetectionResults detections={detections} />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2026 AI Safety Monitoring System | Backend: {status.model_loaded ? '✅ Ready' : '❌ Not Ready'}</p>
      </footer>
    </div>
  );
}

export default App;
