import React, { useState } from 'react';
import { usePollingStable } from '../hooks/useApiStable';
import { apiEndpoints } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import AlertBanner from '../components/AlertBanner';
import { 
  Users, 
  Search, 
  Shield, 
  AlertTriangle,
  CheckCircle,
  User,
  MapPin,
  Clock,
  Trash2
} from 'lucide-react';

const Workers = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  
  const { data: workers, loading, error, refetch } = usePollingStable(apiEndpoints.getWorkers, 10000);

  const handleDeleteWorker = async (workerId) => {
    if (!window.confirm('Are you sure you want to delete this worker? This will also delete all their face profiles and attendance records.')) {
      return;
    }

    try {
      const response = await fetch(`http://localhost:8080/api/attendance/employees/${workerId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        refetch();
      } else {
        alert('Failed to delete worker');
      }
    } catch (error) {
      console.error('Error deleting worker:', error);
      alert('Error deleting worker');
    }
  };

  const filteredWorkers = workers?.workers?.filter(worker => {
    const matchesSearch = (worker.name?.toLowerCase() || '').includes(searchTerm.toLowerCase()) ||
                         (worker.role?.toLowerCase() || '').includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || worker.status === statusFilter;
    return matchesSearch && matchesStatus;
  }) || [];

  const getStatusIcon = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'at risk':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-600" />;
    }
  };

  const getStatusColor = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'at risk':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const getRoleColor = (role) => {
    const colors = {
      'No Helmet': 'bg-red-100 text-red-800',
      'No Vest': 'bg-orange-100 text-orange-800',
      'Safe': 'bg-green-100 text-green-800',
      'Multiple Violations': 'bg-purple-100 text-purple-800',
    };
    return colors[role] || 'bg-gray-100 text-gray-800';
  };

  const activeWorkers = filteredWorkers.filter(w => w.status === 'Active').length;
  const atRiskWorkers = filteredWorkers.filter(w => w.status === 'At Risk').length;

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
        <h1 className="text-3xl font-bold text-gray-900">Employee</h1>
        <p className="text-gray-600 mt-2">Monitor employee safety and status</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Employee</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{filteredWorkers.length}</p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Users className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active</p>
              <p className="text-2xl font-bold text-green-600 mt-1">{activeWorkers}</p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">At Risk</p>
              <p className="text-2xl font-bold text-red-600 mt-1">{atRiskWorkers}</p>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-red-600" />
            </div>
          </div>
        </div>
      </div>

      {error && (
        <AlertBanner 
          type="error" 
          message="Failed to load employee data. Please try again." 
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
                placeholder="Search employee..."
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
              <option value="Active">Active</option>
              <option value="At Risk">At Risk</option>
            </select>
          </div>
        </div>
      </div>

      {/* Workers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredWorkers.length > 0 ? (
          filteredWorkers.map((worker) => (
            <div key={worker.worker_id} className="card hover:shadow-md transition-shadow duration-200">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <div className="p-2 bg-gray-100 rounded-lg">
                    <User className="h-6 w-6 text-gray-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="text-lg font-medium text-gray-900">{worker.name}</h3>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRoleColor(worker.role)}`}>
                      {worker.role}
                    </span>
                  </div>
                </div>
                <div className="flex items-center">
                  {getStatusIcon(worker.status)}
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center text-sm text-gray-600">
                  <MapPin className="h-4 w-4 mr-2" />
                  <span>Location: {worker.name.includes('from') ? worker.name.split('from')[1].trim() : 'Unknown'}</span>
                </div>
                
                <div className="flex items-center text-sm text-gray-600">
                  <Users className="h-4 w-4 mr-2" />
                  <span>Group Size: {worker.persons || 1}</span>
                </div>
                
                <div className="flex items-center text-sm text-gray-600">
                  <Shield className="h-4 w-4 mr-2" />
                  <span>Last Seen: {new Date(worker.timestamp).toLocaleString()}</span>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(worker.status)}`}>
                    {getStatusIcon(worker.status)}
                    <span className="ml-1">{worker.status}</span>
                  </span>
                  <div className="flex items-center gap-2">
                    {worker.status === 'At Risk' && (
                      <button className="text-sm text-primary-600 hover:text-primary-800 font-medium">
                        View Details
                      </button>
                    )}
                    <button
                      onClick={() => handleDeleteWorker(worker.worker_id)}
                      className="text-sm text-red-600 hover:text-red-800 font-medium flex items-center gap-1"
                    >
                      <Trash2 className="h-4 w-4" />
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="col-span-full">
            <div className="text-center py-12">
              <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No employee found</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Workers;
