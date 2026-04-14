import React, { useState, useEffect } from 'react';
import { usePolling } from '../hooks/useApi';
import { apiEndpoints } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import AlertBanner from '../components/AlertBanner';
import { 
  Bell, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Search,
  Filter,
  Volume2,
  VolumeX
} from 'lucide-react';

const Alerts = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [levelFilter, setLevelFilter] = useState('all');
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [dismissedAlerts, setDismissedAlerts] = useState(new Set());
  
  const { data: alerts, loading, error, refetch } = usePolling(apiEndpoints.getAlerts, 5000);

  // Play sound for new high-priority alerts
  useEffect(() => {
    if (soundEnabled && alerts) {
      const newHighAlerts = alerts.alerts?.filter(alert => 
        alert.level === 'high' && !dismissedAlerts.has(alert.id)
      );
      
      if (newHighAlerts.length > 0) {
        // In a real app, you would play an actual sound file
        console.log('🔔 New high-priority alert detected!');
      }
    }
  }, [alerts, soundEnabled, dismissedAlerts]);

  const handleDismiss = (alertId) => {
    setDismissedAlerts(prev => new Set([...prev, alertId]));
  };

  const handleClearAll = () => {
    const allAlertIds = alerts?.alerts?.map(alert => alert.id) || [];
    setDismissedAlerts(new Set(allAlertIds));
  };

  const filteredAlerts = alerts?.alerts?.filter(alert => {
    const matchesSearch = (alert.message?.toLowerCase() || '').includes(searchTerm.toLowerCase());
    const matchesLevel = levelFilter === 'all' || alert.priority === levelFilter;
    const notDismissed = !dismissedAlerts.has(alert.id);
    return matchesSearch && matchesLevel && notDismissed;
  }) || [];

  const getAlertIcon = (level) => {
    switch (level) {
      case 'high':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'low':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
    }
  };

  const getAlertColor = (level) => {
    switch (level) {
      case 'high':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'low':
        return 'bg-green-50 border-green-200 text-green-800';
      default:
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
    }
  };

  const getLevelBadgeColor = (level) => {
    switch (level) {
      case 'high':
        return 'bg-red-100 text-red-800';
      case 'low':
        return 'bg-green-100 text-green-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const highPriorityCount = filteredAlerts.filter(alert => alert.level === 'high').length;
  const lowPriorityCount = filteredAlerts.filter(alert => alert.level === 'low').length;

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
            <h1 className="text-3xl font-bold text-gray-900">Alerts</h1>
            <p className="text-gray-600 mt-2">Real-time safety alerts and notifications</p>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-lg border ${soundEnabled ? 'bg-primary-50 border-primary-200 text-primary-600' : 'bg-gray-50 border-gray-200 text-gray-600'}`}
              title={soundEnabled ? 'Disable sound' : 'Enable sound'}
            >
              {soundEnabled ? <Volume2 className="h-5 w-5" /> : <VolumeX className="h-5 w-5" />}
            </button>
            {filteredAlerts.length > 0 && (
              <button
                onClick={handleClearAll}
                className="btn btn-secondary"
              >
                Clear All
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Alert Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Alerts</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{filteredAlerts.length}</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Bell className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">High Priority</p>
              <p className="text-2xl font-bold text-red-600 mt-1">{highPriorityCount}</p>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <XCircle className="h-6 w-6 text-red-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Low Priority</p>
              <p className="text-2xl font-bold text-green-600 mt-1">{lowPriorityCount}</p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>
      </div>

      {error && (
        <AlertBanner 
          type="error" 
          message="Failed to load alerts. Please try again." 
        />
      )}

      {/* Filters */}
      <div className="card mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search alerts..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-gray-400" />
            <select
              value={levelFilter}
              onChange={(e) => setLevelFilter(e.target.value)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Levels</option>
              <option value="high">High Priority</option>
              <option value="low">Low Priority</option>
            </select>
          </div>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {filteredAlerts.length > 0 ? (
          filteredAlerts.map((alert) => (
            <div 
              key={alert.id} 
              className={`border rounded-lg p-4 transition-all duration-200 ${getAlertColor(alert.level)} ${
                alert.level === 'high' ? 'alert-pulse' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getAlertIcon(alert.level || 'medium')}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getLevelBadgeColor(alert.level)}`}>
                        {alert.level?.toUpperCase() || 'INFO'}
                      </span>
                      <span className="text-xs text-gray-500">
                        ID: #{alert.id || 'N/A'}
                      </span>
                    </div>
                    <p className="text-sm font-medium">{alert.message || 'No message'}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      Status: {alert.status || 'Unknown'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleDismiss(alert.id)}
                    className="text-current hover:opacity-75 transition-opacity"
                    title="Dismiss alert"
                  >
                    <XCircle className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-12">
            <Bell className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No alerts found</p>
            <p className="text-sm text-gray-400 mt-2">Alerts will appear here when safety issues are detected</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Alerts;
