import React, { useState, useEffect } from 'react';
import { Camera, AlertTriangle, Shield, CheckCircle } from 'lucide-react';

const WebcamPermissions = ({ onPermissionGranted, onPermissionDenied }) => {
  const [permissionStatus, setPermissionStatus] = useState('unknown');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    checkPermissions();
  }, []);

  const checkPermissions = async () => {
    try {
      // Check if mediaDevices is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setPermissionStatus('unsupported');
        return;
      }

      // Try to get camera permissions
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false 
      });
      
      // If we get here, permissions are granted
      stream.getTracks().forEach(track => track.stop());
      setPermissionStatus('granted');
      if (onPermissionGranted) onPermissionGranted();
      
    } catch (error) {
      console.error('Permission check failed:', error);
      
      if (error.name === 'NotAllowedError') {
        setPermissionStatus('denied');
        if (onPermissionDenied) onPermissionDenied();
      } else if (error.name === 'NotFoundError') {
        setPermissionStatus('no-camera');
      } else if (error.name === 'NotReadableError') {
        setPermissionStatus('in-use');
      } else {
        setPermissionStatus('error');
      }
    }
  };

  const requestPermissions = async () => {
    setIsLoading(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }, 
        audio: false 
      });
      
      stream.getTracks().forEach(track => track.stop());
      setPermissionStatus('granted');
      if (onPermissionGranted) onPermissionGranted();
      
    } catch (error) {
      console.error('Permission request failed:', error);
      
      if (error.name === 'NotAllowedError') {
        setPermissionStatus('denied');
        if (onPermissionDenied) onPermissionDenied();
      } else {
        setPermissionStatus('error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusMessage = () => {
    switch (permissionStatus) {
      case 'granted':
        return {
          icon: <CheckCircle className="h-8 w-8 text-green-600" />,
          title: "Camera Permissions Granted",
          description: "You can now use your webcam for safety monitoring.",
          color: 'text-green-600'
        };
      case 'denied':
        return {
          icon: <AlertTriangle className="h-8 w-8 text-red-600" />,
          title: "Camera Permissions Denied",
          description: "Please enable camera permissions in your browser settings.",
          color: 'text-red-600'
        };
      case 'no-camera':
        return {
          icon: <Camera className="h-8 w-8 text-gray-600" />,
          title: "No Camera Found",
          description: "Please connect a camera to your device.",
          color: 'text-gray-600'
        };
      case 'in-use':
        return {
          icon: <AlertTriangle className="h-8 w-8 text-yellow-600" />,
          title: "Camera In Use",
          description: "Camera is being used by another application.",
          color: 'text-yellow-600'
        };
      case 'unsupported':
        return {
          icon: <AlertTriangle className="h-8 w-8 text-red-600" />,
          title: "Browser Not Supported",
          description: "Your browser doesn't support camera access.",
          color: 'text-red-600'
        };
      case 'error':
        return {
          icon: <AlertTriangle className="h-8 w-8 text-red-600" />,
          title: "Camera Error",
          description: "An error occurred while accessing the camera.",
          color: 'text-red-600'
        };
      default:
        return {
          icon: <Shield className="h-8 w-8 text-blue-600" />,
          title: "Checking Camera Permissions",
          description: "Please wait while we check your camera permissions...",
          color: 'text-blue-600'
        };
    }
  };

  const status = getStatusMessage();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-8 max-w-md w-full mx-4 shadow-2xl">
        <div className="text-center">
          <div className="mb-4 flex justify-center">
            {status.icon}
          </div>
          
          <h2 className={`text-xl font-semibold mb-2 ${status.color}`}>
            {status.title}
          </h2>
          
          <p className="text-gray-600 mb-6">
            {status.description}
          </p>

          {permissionStatus === 'denied' && (
            <div className="text-left bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-gray-900 mb-2">How to enable camera permissions:</h3>
              <ol className="text-sm text-gray-600 space-y-2">
                <li>1. Click the camera icon 📷 in your browser's address bar</li>
                <li>2. Select "Allow" for camera access</li>
                <li>3. Refresh the page and try again</li>
              </ol>
              <p className="text-xs text-gray-500 mt-3">
                Note: Make sure no other app is using your camera
              </p>
            </div>
          )}

          {permissionStatus === 'unsupported' && (
            <div className="text-left bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-gray-900 mb-2">Browser Requirements:</h3>
              <p className="text-sm text-gray-600">
                Please use a modern browser like Chrome, Firefox, Safari, or Edge.
              </p>
            </div>
          )}

          {permissionStatus === 'no-camera' && (
            <div className="text-left bg-gray-50 rounded-lg p-4 mb-6">
              <h3 className="font-medium text-gray-900 mb-2">Camera Check:</h3>
              <p className="text-sm text-gray-600">
                Please ensure your webcam is properly connected and not disabled in device settings.
              </p>
            </div>
          )}

          <div className="flex space-x-3 justify-center">
            {permissionStatus !== 'granted' && (
              <button
                onClick={requestPermissions}
                disabled={isLoading}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2" />
                    Requesting...
                  </div>
                ) : (
                  'Enable Camera'
                )}
              </button>
            )}
            
            {permissionStatus === 'granted' && (
              <button
                onClick={() => window.location.reload()}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Continue to Webcam
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebcamPermissions;
