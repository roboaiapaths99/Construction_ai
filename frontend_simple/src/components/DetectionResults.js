import React from 'react';
import './DetectionResults.css';

function DetectionResults({ detections }) {
  return (
    <div className="detection-results">
      <h2>📊 Detections</h2>
      <div className="results-list">
        {detections && detections.length > 0 ? (
          <div>
            <div className="detection-count">
              {detections.length} object{detections.length !== 1 ? 's' : ''} detected
            </div>
            {detections.map((detection, idx) => (
              <div key={idx} className="detection-item">
                <div className="detection-header">
                  <span className="class-name">{detection.class_name}</span>
                  <span className="confidence">
                    {(detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="detection-details">
                  <small>Class ID: {detection.class_id}</small>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-detections">No detections yet</div>
        )}
      </div>
    </div>
  );
}

export default DetectionResults;
