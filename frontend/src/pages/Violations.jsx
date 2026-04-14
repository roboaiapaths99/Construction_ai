import React, { useState } from 'react';
import { useApi } from '../hooks/useApi';
import { apiEndpoints } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import AlertBanner from '../components/AlertBanner';
import Pagination from '../components/Pagination';
import { useToast } from '../context/ToastContext';
import { exportToCSV, generateReport } from '../utils/exportUtils';
import { 
  AlertTriangle, 
  Search, 
  Filter, 
  Eye,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  FileText
} from 'lucide-react';

const ITEMS_PER_PAGE = 10;

const Violations = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedViolation, setSelectedViolation] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const toast = useToast();
  
  const { data: violations, loading, error, refetch } = useApi(apiEndpoints.getViolations);

  const handleStatusUpdate = async (violationId, newStatus) => {
    try {
      await apiEndpoints.updateIncidentStatus(violationId, newStatus);
      refetch();
      setSelectedViolation(null);
      toast.success('Violation status updated successfully');
    } catch (error) {
      console.error('Failed to update status:', error);
      toast.error('Failed to update violation status');
    }
  };

  const filteredViolations = violations?.violations?.filter(violation => {
    const matchesSearch = (violation.violation_type?.toLowerCase() || '').includes(searchTerm.toLowerCase()) ||
                         (violation.camera_name?.toLowerCase() || '').includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || violation.status === statusFilter;
    return matchesSearch && matchesStatus;
  }) || [];

  // Pagination
  const totalPages = Math.ceil(filteredViolations.length / ITEMS_PER_PAGE);
  const paginatedViolations = filteredViolations.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  const handleExportCSV = () => {
    try {
      const exportData = filteredViolations.map(v => ({
        'Violation Type': v.violation_type,
        'Camera': v.camera_name,
        'Workers': v.persons,
        'Time': new Date(v.timestamp).toLocaleString(),
        'Status': v.status,
        'Confidence': `${(v.confidence * 100).toFixed(2)}%`,
      }));
      exportToCSV(exportData, 'violations');
      toast.success('Violations exported to CSV');
    } catch (error) {
      toast.error('Failed to export violations');
    }
  };

  const handleGenerateReport = () => {
    try {
      const sections = [
        {
          title: 'Summary',
          content: `Total Violations: ${filteredViolations.length}
Open: ${filteredViolations.filter(v => v.status === 'open').length}
Resolved: ${filteredViolations.filter(v => v.status === 'resolved').length}`,
        },
        {
          title: 'Violations Details',
          content: filteredViolations,
        },
      ];
      generateReport('Violations Report', sections);
      toast.success('Report generated');
    } catch (error) {
      toast.error('Failed to generate report');
    }
  };

  const getStatusIcon = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'resolved':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'open':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-600" />;
    }
  };

  const getStatusColor = (status) => {
    switch ((status || '').toLowerCase()) {
      case 'resolved':
        return 'bg-green-100 text-green-800';
      case 'open':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

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
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Violations</h1>
            <p className="text-gray-600 mt-2">Monitor and manage safety violations</p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleExportCSV}
              disabled={filteredViolations.length === 0}
              className="btn btn-secondary flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Export CSV
            </button>
            <button
              onClick={handleGenerateReport}
              disabled={filteredViolations.length === 0}
              className="btn btn-secondary flex items-center gap-2"
            >
              <FileText className="h-4 w-4" />
              Report
            </button>
          </div>
        </div>
      </div>

      {error && (
        <AlertBanner 
          type="error" 
          message="Failed to load violations. Please try again." 
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
                placeholder="Search violations..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-gray-400" />
            <select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setCurrentPage(1);
              }}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Status</option>
              <option value="open">Open</option>
              <option value="resolved">Resolved</option>
              <option value="investigating">Investigating</option>
            </select>
          </div>
        </div>
      </div>

      {/* Violations List */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Camera
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Workers
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {paginatedViolations.length > 0 ? (
                paginatedViolations.map((violation) => (
                  <tr key={violation.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <AlertTriangle className="h-5 w-5 text-red-600 mr-2" />
                        <span className="text-sm font-medium text-gray-900">{violation.violation_type}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {violation.camera_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {violation.persons || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(violation.timestamp).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(violation.status)}`}>
                        {getStatusIcon(violation.status)}
                        <span className="ml-1">{violation.status}</span>
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => setSelectedViolation(violation)}
                        className="text-primary-600 hover:text-primary-900 mr-3"
                        title="View details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      {violation.status === 'open' && (
                        <button
                          onClick={() => handleStatusUpdate(violation.id, 'resolved')}
                          className="text-green-600 hover:text-green-900"
                          title="Mark as resolved"
                        >
                          <CheckCircle className="h-4 w-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                    No violations found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {paginatedViolations.length > 0 && (
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={setCurrentPage}
            itemsPerPage={ITEMS_PER_PAGE}
            totalItems={filteredViolations.length}
            isLoading={loading}
          />
        )}
      </div>

      {/* Violation Detail Modal */}
      {selectedViolation && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-lg bg-white">
            <div className="mt-3">
              <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                Violation Details
              </h3>
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-600">Type</p>
                  <p className="font-medium">{selectedViolation.violation_type}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Camera</p>
                  <p className="font-medium">{selectedViolation.camera_name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Time</p>
                  <p className="font-medium">{new Date(selectedViolation.timestamp).toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Confidence</p>
                  <p className="font-medium">{(selectedViolation.confidence * 100).toFixed(2)}%</p>
                </div>
              </div>
              <div className="mt-4 flex gap-3">
                <button
                  onClick={() => setSelectedViolation(null)}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-900 rounded-lg hover:bg-gray-300"
                >
                  Close
                </button>
                {selectedViolation.status === 'open' && (
                  <button
                    onClick={() => {
                      handleStatusUpdate(selectedViolation.id, 'resolved');
                    }}
                    className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
                  >
                    Resolve
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Violations;
