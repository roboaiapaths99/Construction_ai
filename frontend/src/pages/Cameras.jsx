import React, { useState, useEffect } from 'react';
import { usePollingStable } from '../hooks/useApiStable';
import { apiEndpoints } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import AlertBanner from '../components/AlertBanner';
import WebcamAI from '../components/WebcamAI';
import { 
  Camera, 
  Search, 
  Settings,
  Eye,
  EyeOff,
  Wifi,
  WifiOff,
  Video,
  VideoOff,
  RefreshCw,
  Monitor,
  CameraOff
} from 'lucide-react';

const Cameras = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [showWebcam, setShowWebcam] = useState(false);
  const [showRTSPCamera, setShowRTSPCamera] = useState(false);
  const [selectedRTSPCamera, setSelectedRTSPCamera] = useState(null);
  const [webcamView, setWebcamView] = useState('grid'); // 'grid' or 'fullscreen'
  
  const { data: cameras, loading, error, refetch, setWebcamActive } = usePollingStable(
    apiEndpoints.getCameras, 
    30000, // 30 seconds instead of 10
    [], 
    true // prevent rerender when webcam is active
  );

  // Mock camera status data - in real app, this would come from backend
  const [cameraStatuses, setCameraStatuses] = useState({
    1: { online: true, recording: true, lastSeen: new Date() },
    2: { online: true, recording: false, lastSeen: new Date() },
    3: { online: false, recording: false, lastSeen: new Date(Date.now() - 300000) },
    4: { online: true, recording: true, lastSeen: new Date() },
  });

  // Track webcam state for polling
  useEffect(() => {
    setWebcamActive(showWebcam);
  }, [showWebcam, setWebcamActive]);

  const filteredCameras = cameras?.cameras?.filter(camera => {
    const matchesSearch = camera.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         camera.location.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || 
                         (statusFilter === 'online' && cameraStatuses[camera.id]?.online) ||
                         (statusFilter === 'offline' && !cameraStatuses[camera.id]?.online);
    return matchesSearch;
  }) || [];

  const toggleCameraRecording = async (cameraId) => {
    setCameraStatuses(prev => ({
      ...prev,
      [cameraId]: {
        ...prev[cameraId],
        recording: !prev[cameraId]?.recording
      }
    }));
  };

  const getStatusColor = (online) => {
    return online ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  const getRecordingColor = (recording) => {
    return recording ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800';
  };

  const onlineCameras = filteredCameras.filter(c => cameraStatuses[c.id]?.online).length;
  const recordingCameras = filteredCameras.filter(c => cameraStatuses[c.id]?.recording).length;

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Cameras</h1>
            <p className="text-gray-600 mt-2">Monitor and manage surveillance cameras</p>
          </div>
          <button
            onClick={refetch}
            className="btn btn-secondary flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Camera Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Cameras</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{filteredCameras.length}</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Camera className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Online</p>
              <p className="text-2xl font-bold text-green-600 mt-1">{onlineCameras}</p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <Wifi className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Recording</p>
              <p className="text-2xl font-bold text-blue-600 mt-1">{recordingCameras}</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Video className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>
      </div>

      {error && (
        <AlertBanner 
          type="error" 
          message="Failed to load camera data. Please try again." 
        />
      )}

      {/* Webcam Toggle */}
      <div className="card mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Local Webcam</h3>
            <p className="text-sm text-gray-600">Use your laptop camera for live monitoring</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setWebcamView(webcamView === 'grid' ? 'fullscreen' : 'grid')}
              className={`btn ${webcamView === 'fullscreen' ? 'btn-secondary' : 'btn-primary'}`}
              disabled={!showWebcam}
            >
              <Monitor className="h-4 w-4 mr-2" />
              {webcamView === 'fullscreen' ? 'Grid View' : 'Fullscreen'}
            </button>
            <button
              onClick={() => setShowWebcam(!showWebcam)}
              className={`btn ${showWebcam ? 'btn-danger' : 'btn-primary'}`}
            >
              {showWebcam ? (
                <>
                  <CameraOff className="h-4 w-4 mr-2" />
                  Hide Webcam
                </>
              ) : (
                <>
                  <Camera className="h-4 w-4 mr-2" />
                  Show Webcam
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Webcam Display */}
      {showWebcam && (
        <div className={`mb-6 ${webcamView === 'fullscreen' ? 'col-span-full' : ''}`}>
          <div className="card">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">🤖 AI-Powered Live Detection</h3>
              <p className="text-sm text-gray-600">Real-time object detection with YOLO AI model</p>
            </div>
            <div className={`${webcamView === 'fullscreen' ? 'h-96' : 'h-64'}`}>
              <WebcamAI 
                enableRecording={true}
                enableAI={true}
                onDetection={(detectionResult) => {
                  console.log('🤖 AI Detection Result:', detectionResult);
                  // Handle real AI detection results
                  if (detectionResult.violations && detectionResult.violations.length > 0) {
                    console.log('⚠️ Violations detected:', detectionResult.violations);
                  }
                }}
                className="w-full h-full"
              />
            </div>
          </div>
        </div>
      )}

      {/* RTSP Camera View */}
      {showRTSPCamera && selectedRTSPCamera && (
        <div className={`mb-6 ${webcamView === 'fullscreen' ? 'col-span-full' : ''}`}>
          <div className="card">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">📹 IP Camera Stream</h3>
              <p className="text-sm text-gray-600">{selectedRTSPCamera.name} - {selectedRTSPCamera.location}</p>
            </div>
            <div className={`${webcamView === 'fullscreen' ? 'h-96' : 'h-64'}`}>
              <img 
                src={`http://localhost:8001/video_feed`}
                alt={selectedRTSPCamera.name}
                className="w-full h-full object-cover"
                style={{ minHeight: '300px' }}
                onError={(e) => {
                  e.target.style.display = 'none';
                  console.error('Failed to load RTSP stream');
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="card mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search cameras..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Status</option>
              <option value="online">Online</option>
              <option value="offline">Offline</option>
            </select>
          </div>
        </div>
      </div>

      {/* Cameras Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredCameras.length > 0 ? (
          filteredCameras.map((camera) => {
            const status = cameraStatuses[camera.id] || { online: false, recording: false, lastSeen: new Date() };
            
            return (
              <div key={camera.id} className="card hover:shadow-md transition-shadow duration-200">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    <div className="p-2 bg-gray-100 rounded-lg">
                      {status.online ? (
                        <Camera className="h-6 w-6 text-green-600" />
                      ) : (
                        <Camera className="h-6 w-6 text-gray-400" />
                      )}
                    </div>
                    <div className="ml-3">
                      <h3 className="text-lg font-medium text-gray-900">{camera.name}</h3>
                      <p className="text-sm text-gray-500">{camera.ip}</p>
                    </div>
                  </div>
                  <button className="text-gray-400 hover:text-gray-600">
                    <Settings className="h-5 w-5" />
                  </button>
                </div>
                
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Status</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(status.online)}`}>
                      {status.online ? (
                        <>
                          <Wifi className="h-3 w-3 mr-1" />
                          Online
                        </>
                      ) : (
                        <>
                          <WifiOff className="h-3 w-3 mr-1" />
                          Offline
                        </>
                      )}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Recording</span>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRecordingColor(status.recording)}`}>
                      {status.recording ? (
                        <>
                          <Video className="h-3 w-3 mr-1" />
                          Recording
                        </>
                      ) : (
                        <>
                          <VideoOff className="h-3 w-3 mr-1" />
                          Stopped
                        </>
                      )}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Last Seen</span>
                    <span className="text-sm text-gray-900">
                      {new Date(status.lastSeen).toLocaleString()}
                    </span>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    {camera.type === 'rtsp' ? (
                      <button 
                        onClick={() => {
                          setShowRTSPCamera(true);
                          setSelectedRTSPCamera(camera);
                          setShowWebcam(false);
                        }}
                        className="text-sm text-primary-600 hover:text-primary-800 font-medium flex items-center"
                      >
                        <Monitor className="h-4 w-4 mr-1" />
                        View Stream
                      </button>
                    ) : (
                      <>
                        <button className="text-sm text-primary-600 hover:text-primary-800 font-medium flex items-center">
                          <Eye className="h-4 w-4 mr-1" />
                          Live View
                        </button>
                        <button
                          onClick={() => toggleCameraRecording(camera.id)}
                          className={`text-sm font-medium flex items-center ${
                            status.recording 
                              ? 'text-red-600 hover:text-red-800' 
                              : 'text-green-600 hover:text-green-800'
                          }`}
                      disabled={!status.online}
                    >
                      {status.recording ? (
                        <>
                          <VideoOff className="h-4 w-4 mr-1" />
                          Stop
                        </>
                      ) : (
                        <>
                          <Video className="h-4 w-4 mr-1" />
                          Start
                        </>
                      )}
                    </button>
                    </>
                  )}
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className="col-span-full">
            <div className="text-center py-12">
              <Camera className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No cameras found</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Cameras;
