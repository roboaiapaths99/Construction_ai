import React, { useState, useEffect, useRef } from 'react';
import './CameraFeed.css';

function CameraFeed({ isRunning }) {
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(false);
  const frameInterval = useRef(null);

  useEffect(() => {
    if (isRunning) {
      setLoading(true);
      fetchFrame();
      frameInterval.current = setInterval(fetchFrame, 500);
    } else {
      if (frameInterval.current) clearInterval(frameInterval.current);
      setFrame(null);
    }

    return () => {
      if (frameInterval.current) clearInterval(frameInterval.current);
    };
  }, [isRunning]);

  const fetchFrame = async () => {
    try {
      const response = await fetch('http://localhost:8000/camera/frame');
      if (response.ok) {
        const data = await response.json();
        setFrame(data.frame);
        setLoading(false);
      }
    } catch (err) {
      console.error('Failed to fetch frame:', err);
    }
  };

  return (
    <div className="camera-feed">
      <h2>📹 Live Camera Feed</h2>
      <div className="feed-container">
        {frame ? (
          <img 
            src={`data:image/jpeg;base64,${frame}`} 
            alt="Camera Feed"
            className="feed-image"
          />
        ) : (
          <div className="feed-placeholder">
            {loading ? 'Loading...' : isRunning ? 'Fetching frame...' : 'Camera not running'}
          </div>
        )}
      </div>
    </div>
  );
}

export default CameraFeed;
