import React from 'react';
import { usePollingStable } from '../hooks/useApiStable';
import { apiEndpoints } from '../api';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import AlertBanner from '../components/AlertBanner';
import { 
  Users, 
  AlertTriangle, 
  Bell, 
  Camera, 
  TrendingUp,
  Activity,
  Eye
} from 'lucide-react';

const Dashboard = () => {
  const { data: stats, loading: statsLoading, error: statsError } = usePollingStable(
    apiEndpoints.getDashboardStats,
    15000 // Refresh every 15 seconds (was 5)
  );

  const { data: alerts, loading: alertsLoading } = usePollingStable(
    apiEndpoints.getAlerts,
    20000 // Refresh every 20 seconds (was 10)
  );

  const { data: violations, loading: violationsLoading } = usePollingStable(
    apiEndpoints.getViolations,
    25000 // Refresh every 25 seconds (was 15)
  );

  if (statsLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (statsError) {
    return (
      <AlertBanner 
        type="error" 
        message="Failed to load dashboard data. Please check your connection." 
      />
    );
  }

  const recentViolations = violations?.violations?.slice(0, 5) || [];
  const activeAlerts = alerts?.alerts?.filter(alert => alert.level === 'high').slice(0, 3) || [];

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">Real-time construction site safety monitoring</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Workers"
          value={stats?.total_workers || 0}
          icon={Users}
          color="blue"
          trend={5}
        />
        <StatCard
          title="Total Violations"
          value={stats?.total_violations || 0}
          icon={AlertTriangle}
          color="red"
          trend={-2}
        />
        <StatCard
          title="Active Alerts"
          value={stats?.active_alerts || 0}
          icon={Bell}
          color="yellow"
          trend={0}
        />
        <StatCard
          title="Connected Cameras"
          value={stats?.connected_cameras || 0}
          icon={Camera}
          color="green"
          trend={1}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Violations */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Recent Violations</h2>
            <Eye className="h-5 w-5 text-gray-400" />
          </div>
          {violationsLoading ? (
            <LoadingSpinner />
          ) : recentViolations.length > 0 ? (
            <div className="space-y-3">
              {recentViolations.map((violation) => (
                <div key={violation.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">{violation.type}</p>
                    <p className="text-sm text-gray-600">{violation.camera_name} • {new Date(violation.timestamp).toLocaleTimeString()}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    violation.status === 'open' 
                      ? 'bg-red-100 text-red-800' 
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {violation.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No violations recorded</p>
          )}
        </div>

        {/* Active Alerts */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">Active Alerts</h2>
            <Activity className="h-5 w-5 text-gray-400" />
          </div>
          {alertsLoading ? (
            <LoadingSpinner />
          ) : activeAlerts.length > 0 ? (
            <div className="space-y-3">
              {activeAlerts.map((alert) => (
                <div key={alert.id} className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center">
                    <AlertTriangle className="h-5 w-5 text-red-600 mr-2 alert-pulse" />
                    <div className="flex-1">
                      <p className="font-medium text-red-900">{alert.message}</p>
                      <p className="text-sm text-red-700">Level: {alert.level}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No active alerts</p>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="mt-6 card">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
            <p className="text-sm text-gray-600 mt-1">All systems operational</p>
          </div>
          <div className="flex items-center">
            <div className="h-3 w-3 bg-green-500 rounded-full mr-2"></div>
            <span className="text-sm font-medium text-green-700">Online</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
