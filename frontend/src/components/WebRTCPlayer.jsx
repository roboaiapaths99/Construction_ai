import React, { useEffect, useRef, useState } from 'react';
import DetectionOverlay from './DetectionOverlay';

const WebRTCPlayer = ({ streamPath = 'construction_camera', mediaMTXUrl = 'http://localhost:8889', onError, enableDetection = true }) => {
  const videoRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let peerConnection = null;
    let videoElement = null;

    const connectWebRTC = async () => {
      try {
        videoElement = videoRef.current;
        if (!videoElement) return;

        // Get WebRTC offer from MediaMTX using WHEP (WebRTC-HTTP Egress Protocol)
        const response = await fetch(`${mediaMTXUrl}/${streamPath}/whep`, {
          method: 'GET',
        });

        if (!response.ok) {
          throw new Error(`Failed to get WebRTC offer: ${response.statusText}`);
        }

        const offer = await response.text();

        // Create RTCPeerConnection
        peerConnection = new RTCPeerConnection({
          iceServers: [
            { urls: 'stun:stun.l.google.com:19302' }
          ]
        });

        // Handle incoming tracks
        peerConnection.ontrack = (event) => {
          if (event.track.kind === 'video') {
            videoElement.srcObject = event.streams[0];
            setIsConnected(true);
          }
        };

        peerConnection.oniceconnectionstatechange = () => {
          if (peerConnection.iceConnectionState === 'connected') {
            setIsConnected(true);
          } else if (peerConnection.iceConnectionState === 'disconnected' || 
                     peerConnection.iceConnectionState === 'failed') {
            setIsConnected(false);
          }
        };

        peerConnection.onerror = (err) => {
          setError(err);
          if (onError) onError(err);
        };

        // Set remote description (offer)
        await peerConnection.setRemoteDescription(new RTCSessionDescription({
          type: 'offer',
          sdp: offer
        }));

        // Create answer
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);

        // Send answer to MediaMTX using WHEP
        const answerResponse = await fetch(`${mediaMTXUrl}/${streamPath}/whep`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/sdp'
          },
          body: answer.sdp
        });

        if (!answerResponse.ok) {
          throw new Error(`Failed to send answer: ${answerResponse.statusText}`);
        }

      } catch (err) {
        console.error('WebRTC connection error:', err);
        setError(err.message);
        if (onError) onError(err);
      }
    };

    connectWebRTC();

    return () => {
      if (peerConnection) {
        peerConnection.close();
      }
    };
  }, [streamPath, mediaMTXUrl, onError]);

  return (
    <div className="relative w-full h-full bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />
      {enableDetection && isConnected && (
        <DetectionOverlay videoRef={videoRef} />
      )}
      {!isConnected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="text-white text-center">
            {error ? (
              <div className="text-red-500">Error: {error}</div>
            ) : (
              <div className="animate-pulse">Connecting to camera...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default WebRTCPlayer;
