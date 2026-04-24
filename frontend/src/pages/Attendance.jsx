import React, { useEffect, useState } from 'react';
import {
  Camera,
  CheckCircle,
  Clock,
  Plus,
  Upload,
  User,
  X,
} from 'lucide-react';
import config from '../config';
import { useToast } from '../context/ToastContext';

const API_BASE_URL = config.api.baseURL;
const POLL_INTERVAL_MS = 10000;

const emptyEnrollmentForm = {
  worker_id: '',
  name: '',
  email: '',
  phone: '',
};

const Attendance = () => {
  const { success: showSuccessToast, error: showErrorToast } = useToast();
  const [activeTab, setActiveTab] = useState('today');
  const [attendanceData, setAttendanceData] = useState([]);
  const [employees, setEmployees] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showEnrollModal, setShowEnrollModal] = useState(false);
  const [activeWorkerAction, setActiveWorkerAction] = useState('');

  useEffect(() => {
    let isMounted = true;

    const load = async (silent = false) => {
      if (!silent && isMounted) {
        setLoading(true);
      }

      if (silent && isMounted) {
        setRefreshing(true);
      }

      try {
        const [attendanceResponse, employeesResponse, statusResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/api/attendance/today`),
          fetch(`${API_BASE_URL}/api/attendance/employees`),
          fetch(`${API_BASE_URL}/api/attendance/status`),
        ]);

        const [attendancePayload, employeesPayload, statusPayload] = await Promise.all([
          attendanceResponse.json(),
          employeesResponse.json(),
          statusResponse.json(),
        ]);

        if (!attendanceResponse.ok) {
          throw new Error(attendancePayload?.error || 'Failed to load attendance records');
        }

        if (!employeesResponse.ok) {
          throw new Error(employeesPayload?.error || 'Failed to load workers');
        }

        if (!statusResponse.ok) {
          throw new Error(statusPayload?.error || 'Failed to load attendance system status');
        }

        if (isMounted) {
          setAttendanceData(attendancePayload?.records || []);
          setEmployees(employeesPayload?.employees || []);
          setSystemStatus(statusPayload || null);
        }
      } catch (error) {
        if (isMounted) {
          showErrorToast(error.message || 'Unable to load attendance data');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
          setRefreshing(false);
        }
      }
    };

    load(false);
    const interval = setInterval(() => load(true), POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [showErrorToast]);

  const refreshData = async () => {
    setRefreshing(true);

    try {
      const [attendanceResponse, employeesResponse, statusResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/attendance/today`),
        fetch(`${API_BASE_URL}/api/attendance/employees`),
        fetch(`${API_BASE_URL}/api/attendance/status`),
      ]);

      const [attendancePayload, employeesPayload, statusPayload] = await Promise.all([
        attendanceResponse.json(),
        employeesResponse.json(),
        statusResponse.json(),
      ]);

      if (!attendanceResponse.ok || !employeesResponse.ok || !statusResponse.ok) {
        throw new Error(
          attendancePayload?.error ||
            employeesPayload?.error ||
            statusPayload?.error ||
            'Failed to refresh attendance data'
        );
      }

      setAttendanceData(attendancePayload?.records || []);
      setEmployees(employeesPayload?.employees || []);
      setSystemStatus(statusPayload || null);
    } catch (error) {
      showErrorToast(error.message || 'Unable to refresh attendance data');
    } finally {
      setRefreshing(false);
    }
  };

  const handleAttendanceAction = async (workerId, eventType) => {
    setActiveWorkerAction(`${workerId}:${eventType}`);

    try {
      const response = await fetch(`${API_BASE_URL}/api/attendance/mark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          employee_id: workerId,
          event_type: eventType,
        }),
      });

      const payload = await response.json();

      if (!response.ok || payload?.success === false) {
        throw new Error(payload?.error || payload?.message || 'Attendance update failed');
      }

      showSuccessToast(
        eventType === 'check_in'
          ? `Checked in ${workerId}`
          : `Checked out ${workerId}`
      );
      await refreshData();
    } catch (error) {
      showErrorToast(error.message || 'Unable to update attendance');
    } finally {
      setActiveWorkerAction('');
    }
  };

  const enrolledCount = systemStatus?.enrolled_workers ?? employees.length;
  const autoMarkedCount = attendanceData.filter(
    (record) => record.detected_by === 'webcam_auto'
  ).length;

  return (
    <div className="p-6">
      <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Attendance</h1>
          <p className="mt-2 text-gray-600">
            Enroll worker photos and let the webcam mark attendance automatically.
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={refreshData}
            disabled={refreshing}
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>
          <button
            type="button"
            onClick={() => setShowEnrollModal(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            Enroll Worker
          </button>
        </div>
      </div>

      <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <StatCard
          icon={<User className="h-5 w-5 text-blue-600" />}
          label="Enrolled Employee"
          value={enrolledCount}
          tone="blue"
        />
        <StatCard
          icon={<CheckCircle className="h-5 w-5 text-green-600" />}
          label="Today Records"
          value={attendanceData.length}
          tone="green"
        />
        <StatCard
          icon={<Camera className="h-5 w-5 text-amber-600" />}
          label="Auto Marked Today"
          value={autoMarkedCount}
          tone="amber"
        />
      </div>

      <div className="mb-6 rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm font-semibold text-gray-900">Recognition Status</p>
            <p className="mt-1 text-sm text-gray-600">
              Camera: <span className="font-medium">{formatStatus(systemStatus?.camera_status)}</span>
              {' · '}
              Face recognition:{' '}
              <span className="font-medium">
                {systemStatus?.face_recognition_ready ? 'Ready' : 'Waiting for enrollment'}
              </span>
              {' · '}
              Loaded embeddings:{' '}
              <span className="font-medium">{systemStatus?.loaded_embeddings ?? 0}</span>
            </p>
          </div>
          <div
            className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${
              systemStatus?.face_recognition_ready
                ? 'bg-green-100 text-green-800'
                : 'bg-amber-100 text-amber-800'
            }`}
          >
            {systemStatus?.face_recognition_ready
              ? 'Auto attendance active'
              : 'Enroll a worker photo to activate'}
          </div>
        </div>
      </div>

      <div className="mb-6 flex gap-4 border-b border-gray-200">
        <button
          type="button"
          onClick={() => setActiveTab('today')}
          className={`border-b-2 px-1 pb-3 text-sm font-medium transition ${
            activeTab === 'today'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          Today's Attendance
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('employees')}
          className={`border-b-2 px-1 pb-3 text-sm font-medium transition ${
            activeTab === 'employees'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          Enrolled Employee
        </button>
      </div>

      {loading ? (
        <div className="rounded-2xl border border-gray-200 bg-white p-10 text-center text-gray-500 shadow-sm">
          Loading attendance data...
        </div>
      ) : activeTab === 'today' ? (
        <TodayAttendanceTable attendanceData={attendanceData} />
      ) : (
        <WorkersGrid
          employees={employees}
          activeWorkerAction={activeWorkerAction}
          onCheckIn={(workerId) => handleAttendanceAction(workerId, 'check_in')}
          onCheckOut={(workerId) => handleAttendanceAction(workerId, 'check_out')}
        />
      )}

      {showEnrollModal && (
        <EnrollModal
          onClose={() => setShowEnrollModal(false)}
          onSuccess={async () => {
            setShowEnrollModal(false);
            await refreshData();
          }}
        />
      )}
    </div>
  );
};

const TodayAttendanceTable = ({ attendanceData }) => {
  if (!attendanceData.length) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white p-10 text-center text-gray-500 shadow-sm">
        <Clock className="mx-auto mb-3 h-12 w-12 text-gray-300" />
        <p className="text-base font-medium text-gray-700">No attendance records for today</p>
        <p className="mt-2 text-sm text-gray-500">
          Employee recognized by the webcam will appear here automatically.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-600">
                Worker
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-600">
                Check In
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-600">
                Check Out
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-600">
                Source
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-600">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {attendanceData.map((record) => (
              <tr key={`${record.worker_id}-${record.check_in || 'open'}`} className="hover:bg-gray-50">
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-full bg-blue-50 p-2">
                      <User className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{record.name || record.worker_id}</p>
                      <p className="text-xs text-gray-500">{record.worker_id}</p>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 text-sm text-gray-600">{formatTime(record.check_in)}</td>
                <td className="px-6 py-4 text-sm text-gray-600">{formatTime(record.check_out)}</td>
                <td className="px-6 py-4">
                  <span
                    className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${
                      record.detected_by === 'webcam_auto'
                        ? 'bg-purple-100 text-purple-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}
                  >
                    {record.detected_by === 'webcam_auto' ? 'Face detected' : 'Manual'}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-600">
                  {typeof record.confidence === 'number'
                    ? `${Math.round(record.confidence * 100)}%`
                    : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const WorkersGrid = ({ employees, activeWorkerAction, onCheckIn, onCheckOut }) => {
  if (!employees.length) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white p-10 text-center text-gray-500 shadow-sm">
        <User className="mx-auto mb-3 h-12 w-12 text-gray-300" />
        <p className="text-base font-medium text-gray-700">No employee enrolled yet</p>
        <p className="mt-2 text-sm text-gray-500">
          Add a employee with a face photo to enable recognition-based attendance.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
      {employees.map((employee) => {
        const checkInBusy = activeWorkerAction === `${employee.worker_id}:check_in`;
        const checkOutBusy = activeWorkerAction === `${employee.worker_id}:check_out`;

        return (
          <div
            key={employee.worker_id}
            className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm transition hover:shadow-md"
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{employee.name}</h3>
                <p className="text-sm text-gray-500">{employee.worker_id}</p>
              </div>
              <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-semibold text-green-800">
                {employee.status || 'active'}
              </span>
            </div>

            <div className="space-y-2 text-sm text-gray-600">
              <p>Email: {employee.email || '--'}</p>
              <p>Phone: {employee.phone || '--'}</p>
              <p>Enrolled: {formatDate(employee.enrollment_date)}</p>
            </div>

            <div className="mt-5 flex gap-3">
              <button
                type="button"
                onClick={() => onCheckIn(employee.worker_id)}
                disabled={checkInBusy || checkOutBusy}
                className="flex-1 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {checkInBusy ? 'Checking in...' : 'Check In'}
              </button>
              <button
                type="button"
                onClick={() => onCheckOut(employee.worker_id)}
                disabled={checkInBusy || checkOutBusy}
                className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {checkOutBusy ? 'Checking out...' : 'Check Out'}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
};

const EnrollModal = ({ onClose, onSuccess }) => {
  const { success: showSuccessToast } = useToast();
  const [formData, setFormData] = useState(emptyEnrollmentForm);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.type.startsWith('image/')) {
      setError('Please select a JPG or PNG image.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError('Image must be smaller than 5 MB.');
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setError('');
  };

  const clearSelectedFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!formData.name.trim()) {
      setError('Worker name is required.');
      return;
    }

    if (!selectedFile) {
      setError('Please upload a clear face photo.');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      const requestBody = new FormData();
      if (formData.worker_id.trim()) {
        requestBody.append('worker_id', formData.worker_id.trim());
      }
      requestBody.append('name', formData.name.trim());
      requestBody.append('email', formData.email.trim());
      requestBody.append('phone', formData.phone.trim());
      requestBody.append('image', selectedFile);

      const response = await fetch(`${API_BASE_URL}/api/attendance/enroll`, {
        method: 'POST',
        body: requestBody,
      });

      const payload = await response.json();

      if (!response.ok || payload?.success === false) {
        throw new Error(payload?.error || payload?.message || 'Enrollment failed');
      }

      showSuccessToast(`Enrolled ${payload.name} (${payload.worker_id})`);
      setFormData(emptyEnrollmentForm);
      clearSelectedFile();
      await onSuccess();
    } catch (submitError) {
      setError(submitError.message || 'Unable to enroll worker');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl bg-white p-6 shadow-xl">
        <div className="mb-5 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Enroll Worker</h2>
            <p className="mt-1 text-sm text-gray-600">
              Upload one clear face photo to enable webcam auto-attendance.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-2 text-gray-500 transition hover:bg-gray-100 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Worker ID</label>
            <input
              type="text"
              value={formData.worker_id}
              onChange={(event) =>
                setFormData((current) => ({ ...current, worker_id: event.target.value }))
              }
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Leave blank to auto-generate"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">Worker Name *</label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(event) =>
                setFormData((current) => ({ ...current, name: event.target.value }))
              }
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
              placeholder="Enter worker name"
            />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium text-gray-700">Email</label>
              <input
                type="email"
                value={formData.email}
                onChange={(event) =>
                  setFormData((current) => ({ ...current, email: event.target.value }))
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="worker@example.com"
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium text-gray-700">Phone</label>
              <input
                type="tel"
                value={formData.phone}
                onChange={(event) =>
                  setFormData((current) => ({ ...current, phone: event.target.value }))
                }
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="+91 98765 43210"
              />
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700">Face Photo *</label>
            <label htmlFor="enroll-file-input" className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 px-4 py-8 text-center transition hover:border-blue-500 hover:bg-blue-50">
              <Upload className="mb-3 h-8 w-8 text-gray-400" />
              <span className="text-sm font-medium text-gray-700">Click to choose a photo</span>
              <span className="mt-1 text-xs text-gray-500">JPG or PNG, up to 5 MB</span>
              <input
                id="enroll-file-input"
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={handleFileSelect}
              />
            </label>

            {previewUrl ? (
              <div className="relative mt-4 overflow-hidden rounded-xl border border-gray-200">
                <img
                  src={previewUrl}
                  alt="Enrollment preview"
                  className="h-64 w-full object-cover"
                />
                <button
                  type="button"
                  onClick={clearSelectedFile}
                  className="absolute right-3 top-3 rounded-full bg-red-600 p-2 text-white transition hover:bg-red-700"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : null}
          </div>

          <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
            <p className="mb-2 text-sm font-semibold text-blue-900">Photo tips</p>
            <ul className="space-y-1 text-xs text-blue-800">
              <li>Face the camera directly with your full face visible.</li>
              <li>Use even lighting and avoid heavy shadows.</li>
              <li>Try one person per photo with no obstructions over the face.</li>
            </ul>
          </div>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {submitting ? 'Enrolling...' : 'Enroll Worker'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

const StatCard = ({ icon, label, value, tone }) => {
  const toneClasses = {
    blue: 'bg-blue-50',
    green: 'bg-green-50',
    amber: 'bg-amber-50',
  };

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{label}</p>
          <p className="mt-2 text-2xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`rounded-xl p-3 ${toneClasses[tone] || 'bg-gray-50'}`}>{icon}</div>
      </div>
    </div>
  );
};

function formatStatus(value) {
  if (!value) {
    return 'Unknown';
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatTime(value) {
  if (!value) {
    return '--';
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return '--';
  }

  return parsed.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatDate(value) {
  if (!value) {
    return '--';
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return '--';
  }

  return parsed.toLocaleDateString([], {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export default Attendance;
