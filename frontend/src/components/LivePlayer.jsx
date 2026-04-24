import React, { useEffect, useRef, useState, useCallback } from "react";
import Hls from 'hls.js';

const LivePlayer = ({ url, hlsUrl, mjpegUrl, cameraStatus }) => {
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [useHls, setUseHls] = useState(true); // Start with HLS (MediaMTX)
  const [useMjpeg, setUseMjpeg] = useState(false); // Don't use MJPEG by default
  const maxRetries = 3;

  const handleStreamError = useCallback(() => {
    setLoading(false);
    if (cameraStatus === "degraded") {
      setError("Camera is not accessible. Check camera network connectivity.");
    } else {
      setError("📡 Stream Not Available - MediaMTX stream endpoint not responding.");
    }
  }, [cameraStatus]);

  useEffect(() => {
    if (!videoRef.current) return;

    setLoading(true);
    setError(null);

    // Cleanup previous HLS instance
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }

    const setupStream = () => {
      if (!videoRef.current) return;
      
      // Try WebRTC first, fallback to HLS
      const streamUrl = useHls ? hlsUrl : url;
      
      if (useHls && hlsUrl) {
        // HLS streaming
        if (Hls.isSupported()) {
          const hls = new Hls({
            debug: true,
            enableWorker: true,
            lowLatencyMode: true,
          });
          hls.loadSource(hlsUrl);
          hls.attachMedia(videoRef.current);
          hlsRef.current = hls;
          
          hls.on(Hls.Events.MANIFEST_PARSED, () => {
            setLoading(false);
            videoRef.current?.play().catch(e => {
              console.error("Play error:", e);
            });
          });
          
          hls.on(Hls.Events.ERROR, (event, data) => {
            console.error("HLS error:", data);
            if (data.fatal) {
              if (retryCount < maxRetries) {
                setRetryCount(retryCount + 1);
                setTimeout(() => {
                  hls.recoverMediaError();
                }, 1000);
              } else {
                handleStreamError();
              }
            }
          });
        } else if (videoRef.current.canPlayType('application/vnd.apple.mpegurl')) {
          // Safari native HLS
          videoRef.current.src = hlsUrl;
          videoRef.current.onloadeddata = () => {
            setLoading(false);
            videoRef.current?.play().catch(e => console.error("Play error:", e));
          };
        } else {
          setError("HLS not supported in this browser");
          setLoading(false);
        }
      } else {
        // WebRTC or MJPEG
        videoRef.current.src = streamUrl;
        videoRef.current.onloadeddata = () => {
          setLoading(false);
          videoRef.current?.play().catch(e => {
            console.error("Play error:", e);
            if (!useHls && hlsUrl) {
              setUseHls(true);
            }
          });
        };
        
        videoRef.current.onerror = () => {
          if (retryCount < maxRetries) {
            setRetryCount(retryCount + 1);
            setTimeout(setupStream, 2000);
          } else if (!useHls && hlsUrl) {
            setUseHls(true);
          } else {
            handleStreamError();
          }
        };
      }
    };

    setupStream();

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [url, hlsUrl, cameraStatus, retryCount, handleStreamError, useHls]);

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          backgroundColor: "#000",
          color: "#fff",
        }}
      >
        <div style={{ fontSize: "48px", marginBottom: "16px" }}>📡</div>
        <h3 style={{ margin: "0 0 8px 0" }}>Stream Not Available</h3>
        <p style={{ margin: "0", fontSize: "14px", color: "#999" }}>{error}</p>
        <p style={{ margin: "8px 0 0 0", fontSize: "12px", color: "#666" }}>
          Using: {useMjpeg ? 'MJPEG' : (useHls ? 'HLS' : 'WebRTC')}
        </p>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      {loading && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1,
          }}
        >
          <div style={{ color: "#fff" }}>Loading stream ({useMjpeg ? 'MJPEG' : useHls ? 'HLS' : 'WebRTC'})...</div>
        </div>
      )}
      {useMjpeg && mjpegUrl ? (
        <img
          src={mjpegUrl}
          alt="Camera Stream"
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
          onLoad={() => setLoading(false)}
          onError={() => {
            if (retryCount < maxRetries) {
              setRetryCount(retryCount + 1);
              setTimeout(() => {
                setUseMjpeg(false);
                setUseHls(true);
              }, 2000);
            } else {
              handleStreamError();
            }
          }}
        />
      ) : (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
          onError={() => {
            if (videoRef.current && retryCount < maxRetries) {
              setRetryCount(retryCount + 1);
              setTimeout(() => {
                if (videoRef.current) {
                  videoRef.current.src = useHls ? hlsUrl : url;
                }
              }, 2000);
            } else if (!useHls && hlsUrl) {
              setUseHls(true);
            } else {
              handleStreamError();
            }
          }}
          onLoadedData={() => setLoading(false)}
        />
      )}
    </div>
  );
};

export default LivePlayer;
