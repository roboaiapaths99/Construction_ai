import React, { useState, useEffect } from 'react';
import { useApi } from '../hooks/useApi';

export default function RecognitionWorkerStatus() {
  const [workerStats, setWorkerStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isRestarting, setIsRestarting] = useState(false);

  const fetchWorkerStats = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/recognition/stats');
      if (!response.ok) throw new Error('Failed to fetch worker stats');
      const data = await response.json();
      setWorkerStats(data);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching worker stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const restartWorker = async () => {
    try {
      setIsRestarting(true);
      const response = await fetch('/api/recognition/restart', {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to restart worker');
      await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2s for restart
      await fetchWorkerStats();
    } catch (err) {
      setError(err.message);
      console.error('Error restarting worker:', err);
    } finally {
      setIsRestarting(false);
    }
  };

  useEffect(() => {
    fetchWorkerStats();
    const interval = setInterval(fetchWorkerStats, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !workerStats) {
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading worker status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800">Recognition Worker Status</h2>
        <button
          onClick={restartWorker}
          disabled={isRestarting}
          className={`px-4 py-2 rounded font-medium transition-colors ${
            isRestarting
              ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600'
          }`}
        >
          {isRestarting ? 'Restarting...' : 'Restart Worker'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-400 rounded text-red-700">
          Error: {error}
        </div>
      )}

      {workerStats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Status */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded border border-blue-200">
            <p className="text-sm text-gray-600 font-medium">Status</p>
            <div className="flex items-center mt-2">
              <div
                className={`w-3 h-3 rounded-full mr-2 ${
                  workerStats.running ? 'bg-green-500' : 'bg-red-500'
                }`}
              ></div>
              <p className="text-lg font-bold">
                {workerStats.running ? 'Running' : 'Stopped'}
              </p>
            </div>
          </div>

          {/* Detector Status */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded border border-purple-200">
            <p className="text-sm text-gray-600 font-medium">Detector</p>
            <div className="flex items-center mt-2">
              <div
                className={`w-3 h-3 rounded-full mr-2 ${
                  workerStats.detector_ready ? 'bg-green-500' : 'bg-yellow-500'
                }`}
              ></div>
              <p className="text-lg font-bold">
                {workerStats.detector_ready ? 'Ready' : 'Loading'}
              </p>
            </div>
          </div>

          {/* Frames Processed */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded border border-green-200">
            <p className="text-sm text-gray-600 font-medium">Frames Processed</p>
            <p className="text-2xl font-bold text-green-700 mt-2">
              {workerStats.frames_processed}
            </p>
          </div>

          {/* Detections */}
          <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded border border-orange-200">
            <p className="text-sm text-gray-600 font-medium">Face Detections</p>
            <p className="text-2xl font-bold text-orange-700 mt-2">
              {workerStats.detections}
            </p>
          </div>
        </div>
      )}

      {workerStats && (
        <div className="mt-4 pt-4 border-t">
          <p className="text-xs text-gray-500">
            Worker ID: <code className="bg-gray-100 px-2 py-1 rounded">{workerStats.worker_id}</code>
          </p>
          <p className="text-xs text-gray-500 mt-2">
            Last Updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
      )}
    </div>
  );
}
