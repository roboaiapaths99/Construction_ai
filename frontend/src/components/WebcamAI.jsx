import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, CameraOff, Video, VideoOff, Download, AlertTriangle, User, Shield, Activity, Zap } from 'lucide-react';
import config from '../config';

const WebcamAI = ({ 
  onDetection, 
  enableRecording = false, 
  enableAI = false,
  className = '',
  showControls = true 
}) => {
  const API_BASE_URL = config.api.baseURL;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectionCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [fps, setFps] = useState(0);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [detections, setDetections] = useState([]);
  const [violations, setViolations] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [debugInfo, setDebugInfo] = useState({
    lastFrameTime: null,
    processingTime: 0,
    apiCalls: 0,
    errors: 0,
    modelLoaded: false
  });

  // Check permissions on mount
  useEffect(() => {
    checkCameraPermissions();
  }, []);

  // Monitor when streaming starts and automatically start AI detection
  useEffect(() => {
    if (isStreaming && enableAI && !detectionIntervalRef.current) {
      console.log('Streaming started, initiating AI detection...');
      setTimeout(() => {
        // Start detection directly without calling startAIDetection to avoid circular dependency
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
        
        console.log('Starting AI detection loop...');
        performAIDetection();
        
        detectionIntervalRef.current = setInterval(() => {
          performAIDetection();
        }, 1500);
      }, 500); // Small delay to ensure video is fully ready
    }
    
    // Cleanup on unmount or when streaming stops
    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    };
  }, [isStreaming, enableAI]);

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
      setDetections([]);
      setViolations([]);
      setDebugInfo(prev => ({ ...prev, apiCalls: 0, errors: 0 }));

      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: enableRecording
      };

      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        videoRef.current.onloadedmetadata = async () => {
          try {
            console.log('Video metadata loaded, starting stream...');
            await videoRef.current.play();
            setIsStreaming(true);
            setPermissionGranted(true);
            
            console.log('Stream started successfully, isStreaming:', true);
            
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
              requestAnimationFrame(calculateFPS);
            };
            
            requestAnimationFrame(calculateFPS);

            console.log('AI detection will start automatically when streaming is ready');
          } catch (error) {
            console.error('Error playing video:', error);
            setError('Failed to start video stream');
          }
        };

        // Also handle oncanplay as backup
        videoRef.current.oncanplay = () => {
          if (!isStreaming) {
            console.log('Video can play, setting streaming true');
            setIsStreaming(true);
          }
        };

        // Handle errors
        videoRef.current.onerror = (error) => {
          console.error('Video error:', error);
          setError('Video stream error occurred');
        };
      }
    } catch (err) {
      console.error('Webcam error:', err);
      
      if (err.name === 'NotAllowedError') {
        setError('Camera permission denied. Please enable camera permissions in your browser.');
        showPermissionInstructions();
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is already in use by another application.');
      } else {
        setError('Failed to access webcam: ' + err.message);
      }
    }
  }, [enableRecording, enableAI]);

  // Stop webcam stream
  const stopWebcam = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
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
    setDetections([]);
    setViolations([]);
    setIsProcessing(false);
  }, []);

  // Enhanced AI Detection with debugging
  const performAIDetection = useCallback(async () => {
    // Enhanced debugging
    const debugInfo = {
      video: !!videoRef.current,
      canvas: !!canvasRef.current,
      streaming: isStreaming,
      processing: isProcessing,
      videoReadyState: videoRef.current?.readyState,
      videoWidth: videoRef.current?.videoWidth,
      videoHeight: videoRef.current?.videoHeight,
      hasSrcObject: !!videoRef.current?.srcObject,
      enableAI: enableAI
    };
    
    console.log('Detection check - conditions:', debugInfo);
    
    if (!videoRef.current || !canvasRef.current || !isStreaming || isProcessing) {
      console.log('Skipping detection - conditions not met:', debugInfo);
      return;
    }
    
    // Additional video readiness check
    if (videoRef.current.readyState < 2) {
      console.log('Video not ready yet, readyState:', videoRef.current.readyState);
      return;
    }

    const startTime = performance.now();
    setIsProcessing(true);
    
    try {
      console.log('Starting AI detection cycle...');
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      // Verify video is ready
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.warn('Video not ready yet, skipping this frame');
        return;
      }

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw current frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to base64
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      console.log(`Frame captured: ${canvas.width}x${canvas.height}, data length: ${imageData.length}`);
      
      // Call backend AI detection API
      const apiStartTime = performance.now();
      const response = await fetch(`${API_BASE_URL}/detect_base64`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData
        })
      });
      
      const apiTime = performance.now() - apiStartTime;
      console.log(`API call completed in ${apiTime.toFixed(2)}ms, status: ${response.status}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('API Response:', result);
      
      // Update debug info
      setDebugInfo(prev => ({
        ...prev,
        lastFrameTime: new Date(),
        processingTime: apiTime,
        apiCalls: prev.apiCalls + 1,
        modelLoaded: result.model && !result.model.includes('Mock')
      }));
      
      if (result.success && result.detections) {
        console.log(`Successful detection: ${result.detections.length} objects, ${result.violations?.length || 0} violations`);
        
        // Process detections with proper validation
        const processedDetections = result.detections.map((detection, index) => {
          // Validate bounding box
          const bbox = detection.bbox || {};
          if (!bbox.x || !bbox.y || !bbox.width || !bbox.height) {
            console.warn(`Invalid bbox for detection ${index}:`, bbox);
            return null;
          }
          
          return {
            id: `detection-${Date.now()}-${index}`,
            class: detection.class_name || detection.class || 'unknown',
            confidence: detection.confidence || 0,
            bbox: {
              x: Math.max(0, bbox.x),
              y: Math.max(0, bbox.y),
              width: Math.max(0, bbox.width),
              height: Math.max(0, bbox.height)
            },
            color: getDetectionColor(detection.class_name || detection.class),
            label: `${(detection.class_name || detection.class || 'UNKNOWN').toUpperCase()} (${Math.round((detection.confidence || 0) * 100)}%)`
          };
        }).filter(Boolean); // Remove null detections

        setDetections(processedDetections);
        
        // Process violations
        if (result.violations && result.violations.length > 0) {
          const processedViolations = result.violations.map((violation, index) => ({
            id: `violation-${Date.now()}-${index}`,
            type: violation.type || 'unknown',
            severity: violation.severity || 'medium',
            confidence: violation.confidence || 0.5,
            bbox: violation.bbox || violation.person_bbox || {},
            message: getViolationMessage(violation.type || 'unknown')
          }));
          
          setViolations(processedViolations);
          console.log(`Violations detected: ${processedViolations.length}`);
        } else {
          setViolations([]);
        }

        // Draw detection boxes
        drawDetectionBoxes(processedDetections, result.violations || []);

        // Callback to parent
        if (onDetection) {
          onDetection({
            detections: processedDetections,
            violations: result.violations || [],
            timestamp: new Date(),
            imageData: imageData,
            stats: {
              totalDetections: processedDetections.length,
              violations: result.violations ? result.violations.length : 0,
              model: result.model || 'Unknown',
              processingTime: result.processing_time_ms || apiTime,
              frameInfo: result.frame_info
            }
          });
        }
      } else if (result.error) {
        console.warn('AI Detection Error:', result.error);
        setDebugInfo(prev => ({ ...prev, errors: prev.errors + 1 }));
        // Continue with mock detection on error
        performMockDetection();
      }

    } catch (error) {
      console.error('AI Detection failed:', error);
      setDebugInfo(prev => ({ ...prev, errors: prev.errors + 1 }));
      // Fallback to mock detection
      performMockDetection();
    } finally {
      const totalTime = performance.now() - startTime;
      setIsProcessing(false);
      console.log(`Detection cycle completed in ${totalTime.toFixed(2)}ms`);
    }
  }, [isStreaming, isProcessing, onDetection]);

  // Enhanced mock detection as fallback
  const performMockDetection = useCallback(() => {
    console.log('Using mock detection fallback');
    
    const mockDetections = [
      {
        id: 'mock-person',
        class: 'person',
        confidence: 0.92,
        bbox: { x: 200, y: 150, width: 120, height: 200 },
        color: '#00FF00',
        label: `PERSON (${Math.round(0.92 * 100)}%)`
      },
      {
        id: 'mock-hardhat',
        class: 'hard_hat',
        confidence: 0.85,
        bbox: { x: 230, y: 120, width: 60, height: 40 },
        color: '#FFD700',
        label: `HARD_HAT (${Math.round(0.85 * 100)}%)`
      },
      {
        id: 'mock-vest',
        class: 'safety_vest',
        confidence: 0.78,
        bbox: { x: 210, y: 190, width: 100, height: 80 },
        color: '#FF69B4',
        label: `SAFETY_VEST (${Math.round(0.78 * 100)}%)`
      }
    ];

    const mockViolations = [
      {
        id: 'violation-1',
        type: 'no_safety_vest',
        severity: 'medium',
        confidence: 0.75,
        bbox: { x: 400, y: 180, width: 100, height: 160 },
        message: 'Worker without safety vest'
      }
    ];

    setDetections(mockDetections);
    setViolations(mockViolations);
    drawDetectionBoxes(mockDetections, mockViolations);

    if (onDetection) {
      onDetection({
        detections: mockDetections,
        violations: mockViolations,
        timestamp: new Date(),
        imageData: canvasRef.current?.toDataURL('image/jpeg', 0.8),
        stats: {
          totalDetections: mockDetections.length,
          violations: mockViolations.length,
          model: 'Mock Detection (Demo)',
          processingTime: 25
        }
      });
    }
  }, [onDetection]);

  // Enhanced detection box drawing
  const drawDetectionBoxes = useCallback((detectionList, violationList) => {
    if (!detectionCanvasRef.current || !videoRef.current) {
      console.warn('Cannot draw boxes - canvas or video not ready');
      return;
    }

    const canvas = detectionCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    console.log(`Drawing ${detectionList.length} detection boxes and ${violationList.length} violation boxes`);

    // Draw detection boxes
    detectionList.forEach((detection, index) => {
      const { bbox, color, label, confidence } = detection;
      
      if (!bbox || bbox.width <= 0 || bbox.height <= 0) {
        console.warn(`Skipping invalid detection box ${index}:`, bbox);
        return;
      }
      
      try {
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
        
        // Draw label background
        ctx.fillStyle = color;
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 20;
        ctx.fillRect(bbox.x, bbox.y - labelHeight, textWidth + 10, labelHeight);
        
        // Draw label text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 12px Arial';
        ctx.fillText(label, bbox.x + 5, bbox.y - 5);
        
        console.log(`Drew detection box: ${detection.class} at (${bbox.x}, ${bbox.y})`);
      } catch (error) {
        console.error(`Error drawing detection box ${index}:`, error);
      }
    });

    // Draw violation boxes (red overlay)
    violationList.forEach((violation, index) => {
      const { bbox, severity, message } = violation;
      
      if (!bbox || bbox.width <= 0 || bbox.height <= 0) {
        console.warn(`Skipping invalid violation box ${index}:`, bbox);
        return;
      }
      
      try {
        // Draw violation box with thicker red border
        ctx.strokeStyle = severity === 'high' ? '#FF0000' : '#FFA500';
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
        ctx.setLineDash([]);
        
        // Draw violation alert background
        ctx.fillStyle = '#FF0000';
        const alertHeight = 35;
        ctx.fillRect(bbox.x, bbox.y - alertHeight, bbox.width, alertHeight);
        
        // Draw violation alert text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 12px Arial';
        ctx.fillText('VIOLATION', bbox.x + 5, bbox.y - 20);
        ctx.font = '10px Arial';
        ctx.fillText(message, bbox.x + 5, bbox.y - 8);
        
        console.log(`Drew violation box: ${violation.type} at (${bbox.x}, ${bbox.y})`);
      } catch (error) {
        console.error(`Error drawing violation box ${index}:`, error);
      }
    });
  }, []);

  // Start AI detection loop (manual trigger)
  const startAIDetection = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }

    console.log('Starting AI detection loop...');
    
    // Initial detection
    performAIDetection();

    // Set up interval for continuous detection
    detectionIntervalRef.current = setInterval(() => {
      performAIDetection();
    }, 1500); // Every 1.5 seconds for smoother detection
  }, []); // Remove performAIDetection dependency to avoid circular reference

  // Helper functions
  const getDetectionColor = (className) => {
    const colors = {
      'person': '#00FF00',      // Green
      'hard_hat': '#FFD700',    // Gold
      'safety_vest': '#FF69B4', // Hot Pink
      'safety_glasses': '#00CED1', // Dark Turquoise
      'gloves': '#FF8C00',      // Dark Orange
      'default': '#FFFFFF'      // White
    };
    return colors[className] || colors.default;
  };

  const getViolationMessage = (violationType) => {
    const messages = {
      'no_hard_hat': 'No Hard Hat',
      'no_safety_vest': 'No Safety Vest',
      'multiple_violations': 'Multiple Violations',
      'unauthorized': 'Unauthorized Person'
    };
    return messages[violationType] || 'Safety Violation';
  };

  // Show browser-specific permission instructions
  const showPermissionInstructions = () => {
    const userAgent = navigator.userAgent;
    const isChrome = /Chrome/.test(userAgent);
    const isFirefox = /Firefox/.test(userAgent);
    const isSafari = /Safari/.test(userAgent) && !/Chrome/.test(userAgent);

    let instructions = '';
    
    if (isChrome) {
      instructions = 'Click the camera icon in the address bar and select "Allow"';
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
      
      {/* Detection Canvas Overlay */}
      <canvas
        ref={detectionCanvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
      
      {/* Hidden canvas for processing */}
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
                  <span className={`px-2 py-1 rounded text-xs ${
                    isProcessing ? 'bg-yellow-600 animate-pulse' : 'bg-blue-600'
                  }`}>
                    AI {isProcessing ? 'Processing...' : 'Active'}
                  </span>
                )}
                {detections.length > 0 && (
                  <span className="bg-purple-600 px-2 py-1 rounded text-xs">
                    {detections.length} Objects
                  </span>
                )}
                {violations.length > 0 && (
                  <span className="bg-red-600 px-2 py-1 rounded text-xs animate-pulse">
                    {violations.length} Violations
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

      {/* AI Detection Stats Overlay */}
      {enableAI && (
        <div className="absolute top-4 left-4 bg-black/80 text-white p-3 rounded-lg max-w-xs">
          <div className="flex items-center mb-2">
            <Zap className="h-4 w-4 mr-2 text-yellow-400" />
            <h4 className="font-semibold text-sm">AI Detection Status</h4>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span>Objects:</span>
              <span className="font-mono">{detections.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Violations:</span>
              <span className={`font-mono ${violations.length > 0 ? 'text-red-400' : 'text-green-400'}`}>
                {violations.length}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Status:</span>
              <span className={`font-mono ${isProcessing ? 'text-yellow-400' : 'text-green-400'}`}>
                {isProcessing ? 'Processing' : 'Ready'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Model:</span>
              <span className={`font-mono ${debugInfo.modelLoaded ? 'text-green-400' : 'text-yellow-400'}`}>
                {debugInfo.modelLoaded ? 'YOLO' : 'Mock'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Calls:</span>
              <span className="font-mono">{debugInfo.apiCalls}</span>
            </div>
            {debugInfo.errors > 0 && (
              <div className="flex justify-between">
                <span>Errors:</span>
                <span className="font-mono text-red-400">{debugInfo.errors}</span>
              </div>
            )}
            {debugInfo.processingTime > 0 && (
              <div className="flex justify-between">
                <span>Time:</span>
                <span className="font-mono">{debugInfo.processingTime.toFixed(1)}ms</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Initial State */}
      {!isStreaming && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
            <Camera className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-400 mb-4">Click to start AI-powered webcam detection</p>
            <button
              onClick={startWebcam}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center mx-auto"
            >
              <Shield className="h-5 w-5 mr-2" />
              Start AI Camera
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default WebcamAI;
