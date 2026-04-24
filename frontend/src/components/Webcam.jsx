import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video, VideoOff, Download, AlertTriangle } from 'lucide-react';
import aiDetectionService from '../services/aiDetection';
import WebcamPermissions from './WebcamPermissions';

const Webcam = ({ 
  onDetection, 
  enableRecording = false, 
  enableAI = false,
  className = '',
  showControls = true 
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState('');
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [detectionResults, setDetectionResults] = useState([]);
  const [fps, setFps] = useState(0);
  const [showPermissionModal, setShowPermissionModal] = useState(false);
  const [permissionGranted, setPermissionGranted] = useState(false);

  // Get available camera devices
  const getDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDevice) {
        setSelectedDevice(videoDevices[0].deviceId);
      }
    } catch (err) {
      setError('Failed to get camera devices');
      console.error('Error getting devices:', err);
    }
  }, [selectedDevice]);

  // Start webcam stream
  const startWebcam = useCallback(async () => {
    try {
      setError('');
      
      // Check permissions first
      if (!permissionGranted) {
        setShowPermissionModal(true);
        return;
      }

      const constraints = {
        video: {
          deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: enableRecording
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        
        // Add event listeners to prevent auto-closing
        stream.getVideoTracks().forEach(track => {
          track.addEventListener('ended', () => {
            console.log('Video track ended');
            // Don't auto-stop, let user control
          });
        });
        
        // Calculate FPS
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
      }
    } catch (err) {
      console.error('Webcam error:', err);
      
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please enable camera permissions in your browser.');
        setShowPermissionModal(true);
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError('Failed to access webcam: ' + err.message);
      }
    }
  }, [selectedDevice, enableRecording, isStreaming, permissionGranted]);

  // Stop webcam stream
  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsRecording(false);
    setFps(0);
    setDetectionResults([]);
  }, []);

  // Start recording
  const startRecording = useCallback(() => {
    if (!streamRef.current) return;

    try {
      chunksRef.current = [];
      const mediaRecorder = new MediaRecorder(streamRef.current, {
        mimeType: 'video/webm;codecs=vp8,opus'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        setRecordedBlob(blob);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('Failed to start recording');
      console.error('Error starting recording:', err);
    }
  }, []);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  // Download recorded video
  const downloadRecording = useCallback(() => {
    if (!recordedBlob) return;

    const url = URL.createObjectURL(recordedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `webcam-recording-${new Date().getTime()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [recordedBlob]);

  // Capture frame for AI detection
  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !enableAI) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 for AI processing
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Perform AI detection
    const detectionResults = await aiDetectionService.detectObjects(imageData);
    
    if (detectionResults) {
      setDetectionResults(detectionResults.detections);
      
      // Report violations to backend
      if (detectionResults.violations.length > 0) {
        for (const violation of detectionResults.violations) {
          try {
            await aiDetectionService.reportViolation(violation, imageData);
          } catch (error) {
            console.error('Failed to report violation:', error);
          }
        }
      }
      
      // Callback for parent component
      if (onDetection) {
        onDetection(detectionResults);
      }
    }
  }, [enableAI, onDetection]);

  // AI detection loop
  useEffect(() => {
    let interval;
    if (isStreaming && enableAI && !isPaused) {
      interval = setInterval(captureFrame, 1000); // Run detection every second
    }
    return () => clearInterval(interval);
  }, [isStreaming, enableAI, isPaused, captureFrame]);

  // Initialize devices on mount
  useEffect(() => {
    getDevices();
    return () => stopWebcam();
  }, [getDevices, stopWebcam]);

  // Check initial permissions
  useEffect(() => {
    const checkInitialPermissions = async () => {
      try {
        const result = await navigator.permissions.query({ name: 'camera' });
        setPermissionGranted(result.state === 'granted');
        
        if (result.state !== 'granted') {
          setShowPermissionModal(true);
        }
      } catch (error) {
        // Some browsers don't support permissions API
        console.log('Permissions API not supported');
      }
    };
    
    checkInitialPermissions();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [stopWebcam, isRecording]);

  return (
    <>
      {/* Permission Modal */}
      {showPermissionModal && (
        <WebcamPermissions
          onPermissionGranted={() => {
            setPermissionGranted(true);
            setShowPermissionModal(false);
          }}
          onPermissionDenied={() => {
            setPermissionGranted(false);
            setShowPermissionModal(false);
          }}
        />
      )}

      {/* Main Webcam Component */}
      <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
        {/* Video Stream */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${isPaused ? 'hidden' : ''}`}
        />
        
        {/* Canvas for AI processing */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Error Display */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
            <div className="text-center p-4">
              <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <p className="text-white text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Detection Overlay */}
        {enableAI && detectionResults.length > 0 && (
          <div className="absolute inset-0 pointer-events-none">
            <svg className="w-full h-full">
              {detectionResults.map((detection, index) => {
                const [x, y, width, height] = detection.bbox;
                const colors = {
                  person: '#10b981',
                  hard_hat: '#3b82f6',
                  safety_vest: '#f59e0b'
                };
                return (
                  <g key={index}>
                    <rect
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill="none"
                      stroke={colors[detection.type] || '#ef4444'}
                      strokeWidth="2"
                    />
                    <text
                      x={x}
                      y={y - 5}
                      fill={colors[detection.type] || '#ef4444'}
                      fontSize="12"
                      fontWeight="bold"
                    >
                      {detection.type} ({Math.round(detection.confidence * 100)}%)
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {/* Controls Overlay */}
        {showControls && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
            <div className="flex items-center justify-between">
              {/* Left Controls */}
              <div className="flex items-center space-x-2">
                {/* Camera Toggle */}
                <button
                  onClick={isStreaming ? stopWebcam : startWebcam}
                  className={`p-2 rounded-full transition-colors ${
                    isStreaming 
                      ? 'bg-red-600 hover:bg-red-700' 
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                  title={isStreaming ? 'Stop Camera' : 'Start Camera'}
                >
                  {isStreaming ? (
                    <CameraOff className="h-5 w-5 text-white" />
                  ) : (
                    <Camera className="h-5 w-5 text-white" />
                  )}
                </button>

                {/* Recording Controls */}
                {enableRecording && isStreaming && (
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

                {/* Pause/Resume */}
                {isStreaming && (
                  <button
                    onClick={() => setIsPaused(!isPaused)}
                    className="p-2 rounded-full bg-gray-600 hover:bg-gray-700 transition-colors"
                    title={isPaused ? 'Resume' : 'Pause'}
                  >
                    {isPaused ? (
                      <Video className="h-5 w-5 text-white" />
                    ) : (
                      <VideoOff className="h-5 w-5 text-white" />
                    )}
                  </button>
                )}
              </div>

              {/* Center Info */}
              <div className="text-white text-sm">
                {isStreaming && (
                  <div className="flex items-center space-x-4">
                    <span className="bg-green-600 px-2 py-1 rounded text-xs">
                      LIVE
                    </span>
                    <span>{fps} FPS</span>
                    {enableAI && (
                      <span className="bg-blue-600 px-2 py-1 rounded text-xs">
                        AI: {detectionResults.length} detections
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Right Controls */}
              <div className="flex items-center space-x-2">
                {/* Device Selector */}
                {devices.length > 1 && (
                  <select
                    value={selectedDevice}
                    onChange={(e) => setSelectedDevice(e.target.value)}
                    className="bg-gray-700 text-white text-sm px-2 py-1 rounded"
                    disabled={isStreaming}
                  >
                    {devices.map(device => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${devices.indexOf(device) + 1}`}
                      </option>
                    ))}
                  </select>
                )}

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
              <p className="text-gray-400 mb-4">Click camera button to start</p>
              <button
                onClick={startWebcam}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Start Webcam
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Webcam;
