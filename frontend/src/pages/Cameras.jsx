import React, { useState, useEffect } from 'react';
import { config } from '../config';
import LivePlayer from '../components/LivePlayer';
import DetectionOverlay from '../components/DetectionOverlay';

const Cameras = () => {
  const [camera, setCamera] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCamera();
    const interval = setInterval(fetchCamera, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchCamera = async () => {
    try {
      const response = await fetch(`${config.api.baseURL}/cameras`);
      const data = await response.json();
      if (data.cameras && data.cameras.length > 0) {
        setCamera(data.cameras[0]);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching camera:', error);
      setLoading(false);
    }
  };

  if (loading) return <div>Loading...</div>;

  if (!camera) {
    return (
      <div style={{ padding: "20px" }}>
        <h2>Site Camera</h2>
        <p style={{ color: "#999" }}>No camera configured</p>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>{camera.name || 'Site Camera'}</h2>

      <div style={{ position: "relative", display: "inline-block", width: "100%" }}>
        {/* 🎥 LIVE VIDEO - Use MediaMTX WebRTC or HLS */}
        <LivePlayer 
          url={config.media.webrtc} 
          hlsUrl={config.media.hls}
          mjpegUrl={config.media.mjpeg}
          cameraStatus={camera.status || 'unknown'} 
        />

        {/* 🧠 AI OVERLAY */}
        <DetectionOverlay detections={detections} />
      </div>

      <div style={{ marginTop: "10px" }}>
        <p>Status: {camera.status || 'unknown'}</p>
        <p>Type: {camera.type || 'ip_camera'}</p>
        <p>Stream URLs:</p>
        <ul>
          <li>WebRTC: {camera.stream_urls?.webrtc || config.media.webrtc}</li>
          <li>HLS: {camera.stream_urls?.hls || config.media.hls}</li>
          <li>RTSP: {camera.stream_urls?.rtsp || config.media.rtsp}</li>
        </ul>
      </div>
    </div>
  );
};

export default Cameras;
