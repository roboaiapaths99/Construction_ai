import React from 'react';
import './StatusPanel.css';

function StatusPanel({ status, onStartCamera, onStopCamera }) {
  return (
    <div className="status-panel">
      <h2>⚙️ Status</h2>
      
      <div className="status-items">
        <div className="status-item">
          <span className="status-label">Model</span>
          <span className={`status-value ${status.model_loaded ? 'status-ok' : 'status-error'}`}>
            {status.model_loaded ? '✅ Loaded' : '❌ Not Ready'}
          </span>
        </div>

        <div className="status-item">
          <span className="status-label">Camera</span>
          <span className={`status-value ${status.camera_running ? 'status-ok' : 'status-error'}`}>
            {status.camera_running ? '✅ Running' : '⏹️ Stopped'}
          </span>
        </div>
      </div>

      <div className="control-section">
        <h3>🎮 Controls</h3>
        <button 
          className="btn-primary" 
          onClick={onStartCamera}
          disabled={status.camera_running}
        >
          Start Camera
        </button>
        <button 
          className="btn-danger" 
          onClick={onStopCamera}
          disabled={!status.camera_running}
        >
          Stop Camera
        </button>
      </div>

      <div className="info-section">
        <h3>ℹ️ Info</h3>
        <p className="info-text">
          Version: 2.0
        </p>
        <p className="info-text">
          Backend: Ready
        </p>
        <p className="info-text">
          Model: YOLOv8n
        </p>
      </div>
    </div>
  );
}

export default StatusPanel;
