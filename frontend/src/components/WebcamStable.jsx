import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video, VideoOff, Download, AlertTriangle } from 'lucide-react';
import { detectObjects } from '../api/detection';

const WebcamStable = ({ 
  onDetection, 
  enableRecording = false, 
  enableAI = false,
  className = '',
  showControls = true 
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recordingRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [fps, setFps] = useState(0);
  const [retryCount, setRetryCount] = useState(0);
  const [streamActive, setStreamActive] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [detectionLog, setDetectionLog] = useState([]);

  // Prevent stream from closing
  const keepStreamActive = useCallback(() => {
    if (streamRef.current && streamActive) {
      // Send a keep-alive signal to prevent auto-closing
      const tracks = streamRef.current.getVideoTracks();
      tracks.forEach(track => {
        if (track.readyState === 'live') {
          // Track is still active
          console.log('Stream track is active:', track.label);
        }
      });
    }
  }, [streamActive]);

  // Monitor stream status
  useEffect(() => {
    const interval = setInterval(() => {
      if (isStreaming && streamRef.current) {
        const tracks = streamRef.current.getVideoTracks();
        const hasActiveTrack = tracks.some(track => track.readyState === 'live');
        
        if (!hasActiveTrack && isStreaming) {
          console.log('Stream lost, attempting to restart...');
          setIsStreaming(false);
          setStreamActive(false);
          // Auto-restart if user didn't manually stop
          if (retryCount < 3) {
            setTimeout(() => {
              startWebcam();
              setRetryCount(prev => prev + 1);
            }, 1000);
          }
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isStreaming, retryCount]);

  // Start webcam stream with enhanced error handling
  const startWebcam = useCallback(async () => {
    try {
      setError('');
      console.log('Starting webcam...');

      // Check if we already have a stream
      if (streamRef.current) {
        const tracks = streamRef.current.getTracks();
        tracks.forEach(track => track.stop());
        streamRef.current = null;
      }

      // Try different constraint combinations
      const constraints = [
        // Ideal constraints
        {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user'
          },
          audio: false
        },
        // Fallback constraints
        {
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
          audio: false
        },
        // Minimal constraints
        {
          video: true,
          audio: false
        }
      ];

      let stream = null;
      let lastError = null;

      for (const constraint of constraints) {
        try {
          console.log('Trying constraints:', constraint);
          stream = await navigator.mediaDevices.getUserMedia(constraint);
          console.log('Stream obtained successfully');
          break;
        } catch (err) {
          console.log('Constraint failed:', err.message);
          lastError = err;
        }
      }

      if (!stream) {
        throw lastError || new Error('Failed to get any stream');
      }

      streamRef.current = stream;
      
      if (videoRef.current) {
        // Set video properties before playing
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.playsInline = true;
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = () => {
          console.log('Video metadata loaded');
          videoRef.current.play().then(() => {
            console.log('Video playing successfully');
            setIsStreaming(true);
            setStreamActive(true);
            setRetryCount(0);
            
            // Start FPS calculation
            let lastTime = performance.now();
            let frames = 0;
            
            const calculateFPS = () => {
              frames++;
              const currentTime = performance.now();
              if (currentTime >= lastTime + 1000) {
                setFps(Math.round((frames * 1000) / (currentTime - lastTime)));
                frames = 0;
                lastTime = currentTime;
              }
              if (isStreaming) {
                requestAnimationFrame(calculateFPS);
              }
            };
            
            requestAnimationFrame(calculateFPS);

            // Log video resolution for debugging
            console.log('Video resolution:', videoRef.current.videoWidth, videoRef.current.videoHeight);

            // Start AI detection if enabled
            if (enableAI) {
              console.log('AI Detection enabled, starting in 1 second...');
              setTimeout(() => {
                if (enableAI && isStreaming) {
                  console.log('Starting detection loop');
                  captureFrame();
                }
              }, 1000);
            }
          }).catch(err => {
            console.error('Video play failed:', err);
            setError('Failed to play video stream');
          });
        };

        // Handle video errors
        videoRef.current.onerror = (err) => {
          console.error('Video error:', err);
          setError('Video stream error occurred');
        };
      }
    } catch (err) {
      console.error('Webcam error:', err);
      
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please click the camera icon 📷 in your browser address bar and select "Allow".');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera and ensure it\'s not being used by another application.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use. Please close other applications using the camera and try again.');
      } else if (err.name === 'OverconstrainedError') {
        setError('Camera does not support the requested settings. Trying lower quality...');
        // Retry with basic constraints
        setTimeout(() => {
          startWebcam();
        }, 1000);
      } else {
        setError(`Camera error: ${err.message || 'Unknown error occurred'}`);
      }
    }
  }, [enableAI, isStreaming]);

  // Stop webcam stream
  const stopWebcam = useCallback(() => {
    console.log('Stopping webcam...');
    
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks();
      tracks.forEach(track => {
        track.stop();
        console.log('Track stopped:', track.label);
      });
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    if (recordingRef.current) {
      recordingRef.current.stop();
    }
    
    setIsStreaming(false);
    setIsRecording(false);
    setStreamActive(false);
    setFps(0);
    setRetryCount(0);
  }, []);

  // Capture frame for AI detection
  const captureFrame = useCallback(async () => {
    console.log('captureFrame called:', { 
      hasVideo: !!videoRef.current, 
      hasCanvas: !!canvasRef.current, 
      enableAI, 
      isStreaming, 
      streamActive,
      videoReady: videoRef.current?.readyState,
      videoWidth: videoRef.current?.videoWidth,
      videoHeight: videoRef.current?.videoHeight
    });
    
    // Temporarily remove streamActive check to force detection
    if (!videoRef.current || !canvasRef.current || !enableAI || !isStreaming) {
      console.log('Skipping capture - basic conditions not met');
      return;
    }

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      console.log('Video readyState:', video.readyState, 'HAVE_ENOUGH_DATA:', video.HAVE_ENOUGH_DATA);

      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        console.log('Canvas drawn, dimensions:', canvas.width, 'x', canvas.height);

        // Convert to base64 and send to backend
        const imageData = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
        console.log('Image data captured, length:', imageData.length);
        console.log('Sending to backend API...');
        
        try {
          const result = await detectObjects(imageData);
          
          if (!result.detections || result.detections.length === 0) {
            setDetectionResults([]);
            
            // Schedule next capture
            if (isStreaming && enableAI) {
              setTimeout(() => {
                captureFrame();
              }, 120);
            }
            return;
          }

          const parsed = result.detections.map(d => {
            const [x1, y1, x2, y2] = d.bbox;
            return {
              type: d.class,
              confidence: d.confidence,
              bbox: [x1, y1, x2 - x1, y2 - y1],
              label: d.class
            };
          });

          console.log('Real detections received:', parsed);
          setDetectionResults(parsed);

          // Add to detection log
          if (parsed.length > 0) {
            const logEntry = {
              timestamp: new Date(),
              detections: parsed,
              violations: []
            };
            setDetectionLog(prev => [logEntry, ...prev.slice(0, 9)]);
            console.log('AI Detection:', parsed);
          }

          // Callback to parent
          if (onDetection) {
            onDetection({
              detections: parsed,
              violations: [],
              timestamp: new Date(),
              imageData: canvas.toDataURL("image/jpeg", 0.8)
            });
          }
        } catch (error) {
          console.error('Backend detection error:', error);
          setDetectionResults([]);
        }
      } else {
        console.log('Video not ready yet, retrying...');
        // Retry after a short delay
        setTimeout(() => {
          if (isStreaming) {
            captureFrame();
          }
        }, 1000);
      }
    } catch (err) {
      console.error('Frame capture error:', err);
    }

    // Schedule next capture - FASTER DETECTION (8 FPS)
    if (isStreaming && enableAI) {
      setTimeout(() => {
        captureFrame();
      }, 120); // Changed from 3000ms to 120ms (8 FPS)
    }
  }, [enableAI, onDetection, isStreaming, streamActive]);

  // Start recording
  const startRecording = useCallback(() => {
    if (!streamRef.current || !enableRecording) return;

    try {
      const options = {
        mimeType: 'video/webm;codecs=vp8'
      };

      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm';
      }

      recordingRef.current = new MediaRecorder(streamRef.current, options);

      const chunks = [];
      
      recordingRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recordingRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        setRecordedBlob(blob);
        setIsRecording(false);
      };

      recordingRef.current.start();
      setIsRecording(true);
      console.log('Recording started');
    } catch (err) {
      console.error('Recording error:', err);
      setError('Failed to start recording');
    }
  }, [enableRecording]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (recordingRef.current && isRecording) {
      recordingRef.current.stop();
      console.log('Recording stopped');
    }
  }, [isRecording]);

  // Download recording
  const downloadRecording = useCallback(() => {
    if (!recordedBlob) return;

    const url = URL.createObjectURL(recordedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `webcam-recording-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [recordedBlob]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, [stopWebcam]);

  // Keep stream active
  useEffect(() => {
    if (isStreaming && streamActive) {
      const interval = setInterval(keepStreamActive, 5000);
      return () => clearInterval(interval);
    }
  }, [isStreaming, streamActive, keepStreamActive]);

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      {/* Video Stream */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />
      
      {/* Canvas for AI processing */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Detection Overlay */}
      {enableAI && detectionResults.length > 0 && (
        <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 10 }}>
          {detectionResults.map((detection, index) => {
            const [x, y, width, height] = detection.bbox;
            const colors = {
              person: '#10b981',
              hard_hat: '#3b82f6',
              safety_vest: '#f59e0b',
              safety_glasses: '#8b5cf6',
              gloves: '#ef4444'
            };
            const color = detection.color || colors[detection.type] || '#ef4444';
            
            // Scale coordinates to fit the actual video display
            const videoElement = videoRef.current;
            const scaleX = videoElement ? videoElement.offsetWidth / (videoElement.videoWidth || 1280) : 0.5;
            const scaleY = videoElement ? videoElement.offsetHeight / (videoElement.videoHeight || 720) : 0.5;
            
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledWidth = Math.max(width * scaleX, 50); // Minimum 50px width
            const scaledHeight = Math.max(height * scaleY, 50); // Minimum 50px height
            
            console.log('Drawing detection:', {
              original: [x, y, width, height],
              scaled: [scaledX, scaledY, scaledWidth, scaledHeight],
              scale: [scaleX, scaleY],
              videoSize: [videoElement?.offsetWidth, videoElement?.offsetHeight],
              videoRes: [videoElement?.videoWidth, videoElement?.videoHeight]
            });
            
            return (
              <div
                key={index}
                className="absolute border-2"
                style={{
                  left: `${scaledX}px`,
                  top: `${scaledY}px`,
                  width: `${scaledWidth}px`,
                  height: `${scaledHeight}px`,
                  borderColor: color,
                  borderStyle: detection.type === 'person' ? 'solid' : 'dashed',
                  borderWidth: '3px',
                  zIndex: 15,
                  backgroundColor: 'transparent',
                  pointerEvents: 'none'
                }}
              >
                {/* Label */}
                <div
                  className="absolute text-white text-xs font-bold"
                  style={{
                    top: '-25px',
                    left: '0px',
                    backgroundColor: color,
                    padding: '2px 6px',
                    borderRadius: '4px',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {detection.label} ({Math.round(detection.confidence * 100)}%)
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Detection Log */}
      {enableAI && detectionLog.length > 0 && (
        <div className="absolute top-4 right-4 bg-black bg-opacity-80 rounded-lg p-2 max-w-xs max-h-32 overflow-y-auto">
          <h4 className="text-white text-xs font-semibold mb-1">AI Detection Log</h4>
          {detectionLog.map((entry, index) => (
            <div key={index} className="text-xs text-gray-300 mb-1">
              <div className="text-blue-400">
                {entry.timestamp.toLocaleTimeString()}
              </div>
              <div>
                {entry.detections.map(d => d.label).join(', ')} 
                {entry.violations.length > 0 && (
                  <span className="text-red-400 ml-1">
                    ⚠️ {entry.violations.length} violation(s)
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center p-4 max-w-md">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-white text-sm mb-4">{error}</p>
            <div className="space-y-2">
              <button
                onClick={startWebcam}
                className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Try Again
              </button>
              <button
                onClick={() => setError('')}
                className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Controls Overlay */}
      {showControls && isStreaming && !error && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
          <div className="flex items-center justify-between">
            {/* Left Controls */}
            <div className="flex items-center space-x-2">
              {/* Camera Toggle */}
              <button
                onClick={stopWebcam}
                className="p-2 rounded-full bg-red-600 hover:bg-red-700 transition-colors"
                title="Stop Camera"
              >
                <CameraOff className="h-5 w-5 text-white" />
              </button>

              {/* Recording Controls */}
              {enableRecording && (
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  className={`p-2 rounded-full transition-colors ${
                    isRecording 
                      ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                      : 'bg-gray-600 hover:bg-gray-700'
                  }`}
                  title={isRecording ? 'Stop Recording' : 'Start Recording'}
                >
                  {isRecording ? (
                    <VideoOff className="h-5 w-5 text-white" />
                  ) : (
                    <Video className="h-5 w-5 text-white" />
                  )}
                </button>
              )}
            </div>

            {/* Center Info */}
            <div className="text-white text-sm">
              <div className="flex items-center space-x-4">
                <span className="bg-green-600 px-2 py-1 rounded text-xs animate-pulse">
                  LIVE
                </span>
                <span>{fps} FPS</span>
                {enableAI && (
                  <span className="bg-blue-600 px-2 py-1 rounded text-xs">
                    AI: {detectionResults.length} objects
                  </span>
                )}
                <span className="text-xs">
                  {streamActive ? 'Stream Active' : 'Stream Inactive'}
                </span>
              </div>
            </div>

            {/* Right Controls */}
            <div className="flex items-center space-x-2">
              {/* Download Recording */}
              {recordedBlob && (
                <button
                  onClick={downloadRecording}
                  className="p-2 rounded-full bg-blue-600 hover:bg-blue-700 transition-colors"
                  title="Download Recording"
                >
                  <Download className="h-5 w-5 text-white" />
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Initial State */}
      {!isStreaming && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
            <Camera className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-400 mb-4">Click to start webcam</p>
            <button
              onClick={startWebcam}
              className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Start Webcam
            </button>
            {retryCount > 0 && (
              <p className="text-yellow-400 text-xs mt-2">
                Retry attempt {retryCount}/3
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamStable;
