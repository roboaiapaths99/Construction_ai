import axios from 'axios';
import config from '../config';

const API_BASE_URL = config.api.baseURL;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// API endpoints
export const apiEndpoints = {
  // Health check
  health: () => api.get('/health'),
  
  // Incidents
  getIncidents: () => api.get('/incidents'),
  getIncident: (id) => api.get(`/incidents/${id}`),
  createIncident: (data) => api.post('/incident', data),
  updateIncidentStatus: (id, status) => api.put(`/incidents/${id}/status`, { status }),
  
  // Dashboard
  getViolations: () => api.get('/violations'),
  getWorkers: () => api.get('/workers'),
  getAlerts: () => api.get('/alerts'),
  getCameras: () => api.get('/cameras'),
  getDashboardStats: () => api.get('/dashboard/stats'),
};

export default api;
