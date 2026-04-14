import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video, VideoOff, Download, AlertTriangle } from 'lucide-react';

const WebcamSimple = ({ 
  onDetection, 
  enableRecording = false, 
  enableAI = false,
  className = '',
  showControls = true 
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [fps, setFps] = useState(0);
  const [permissionGranted, setPermissionGranted] = useState(false);

  // Check permissions on mount
  useEffect(() => {
    checkCameraPermissions();
  }, []);

  const checkCameraPermissions = async () => {
    try {
      const result = await navigator.permissions.query({ name: 'camera' });
      setPermissionGranted(result.state === 'granted');
      
      if (result.state !== 'granted') {
        setError('Camera permission required. Please allow camera access.');
      }
    } catch (error) {
      console.log('Permissions API not supported, will ask on demand');
    }
  };

  // Start webcam stream
  const startWebcam = useCallback(async () => {
    try {
      setError('');

      const constraints = {
        video: {
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
        setPermissionGranted(true);
        
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

        // Start AI detection if enabled
        if (enableAI) {
          setTimeout(() => {
            captureFrame();
          }, 2000); // Start after 2 seconds
        }
      }
    } catch (err) {
      console.error('Webcam error:', err);
      
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please enable camera permissions in your browser.');
        // Show browser-specific instructions
        showPermissionInstructions();
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError('Failed to access webcam: ' + err.message);
      }
    }
  }, [enableRecording, enableAI, isStreaming]);

  // Stop webcam stream
  const stopWebcam = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsRecording(false);
    setFps(0);
  }, []);

  // Capture frame for AI detection
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !enableAI || !isStreaming) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Mock AI detection
    const mockDetections = [
      { 
        type: 'person', 
        confidence: 0.95, 
        bbox: [100, 100, 200, 300],
        label: 'Worker'
      },
      { 
        type: 'hard_hat', 
        confidence: 0.87, 
        bbox: [150, 50, 80, 60],
        label: 'Hard Hat'
      }
    ];

    // Callback to parent
    if (onDetection) {
      onDetection({
        detections: mockDetections,
        violations: [],
        timestamp: new Date(),
        imageData: imageData
      });
    }

    // Schedule next capture
    if (isStreaming && enableAI) {
      setTimeout(() => {
        captureFrame();
      }, 3000); // Every 3 seconds
    }
  }, [enableAI, onDetection, isStreaming]);

  // Show browser-specific permission instructions
  const showPermissionInstructions = () => {
    const userAgent = navigator.userAgent;
    const isChrome = /Chrome/.test(userAgent);
    const isFirefox = /Firefox/.test(userAgent);
    const isSafari = /Safari/.test(userAgent) && !/Chrome/.test(userAgent);

    let instructions = '';
    
    if (isChrome) {
      instructions = 'Click the camera icon 📷 in the address bar and select "Allow"';
    } else if (isFirefox) {
      instructions = 'Click "Remember this decision" when prompted for camera access';
    } else if (isSafari) {
      instructions = 'Go to Safari > Preferences > Websites > Camera and allow this site';
    }

    setError(`Permission needed: ${instructions}`);
  };

  // Start recording
  const startRecording = useCallback(() => {
    if (!streamRef.current || !enableRecording) return;

    try {
      const mediaRecorder = new MediaRecorder(streamRef.current, {
        mimeType: 'video/webm'
      });

      const chunks = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        setRecordedBlob(blob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError('Failed to start recording');
      console.error('Error starting recording:', err);
    }
  }, [enableRecording]);

  // Stop recording
  const stopRecording = useCallback(() => {
    // This would be handled by the MediaRecorder onstop event
    setIsRecording(false);
  }, []);

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
                <span className="bg-green-600 px-2 py-1 rounded text-xs">
                  LIVE
                </span>
                <span>{fps} FPS</span>
                {enableAI && (
                  <span className="bg-blue-600 px-2 py-1 rounded text-xs">
                    AI Active
                  </span>
                )}
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
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamSimple;
