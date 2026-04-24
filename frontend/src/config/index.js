// Environment Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';
const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '10000');
const LOG_LEVEL = process.env.REACT_APP_LOG_LEVEL || 'info';
const ENABLE_DEBUG = process.env.REACT_APP_DEBUG === 'true' || false;
const APP_VERSION = process.env.REACT_APP_VERSION || '1.0.0';
const APP_ENV = process.env.NODE_ENV || 'development';

// MediaMTX Configuration
const MEDIAMTX_WEBRTC_URL = process.env.REACT_APP_MEDIAMTX_WEBRTC || 'http://localhost:8889/sitecam';
const MEDIAMTX_HLS_URL = process.env.REACT_APP_MEDIAMTX_HLS || 'http://localhost:8888/sitecam/index.m3u8';
const MEDIAMTX_RTSP_URL = process.env.REACT_APP_MEDIAMTX_RTSP || 'rtsp://localhost:8554/sitecam';
// Backend MJPEG stream (direct camera feed)
const BACKEND_STREAM_URL = process.env.REACT_APP_BACKEND_STREAM || '/stream';

export const config = {
  api: {
    baseURL: API_BASE_URL,
    timeout: API_TIMEOUT,
    retries: 3,
    retryDelay: 1000,
    isDevelopment: APP_ENV === 'development',
    isProduction: APP_ENV === 'production',
  },
  logging: {
    level: LOG_LEVEL,
    enableDebug: ENABLE_DEBUG,
  },
  app: {
    version: APP_VERSION,
    environment: APP_ENV,
    isDevelopment: APP_ENV === 'development',
    isProduction: APP_ENV === 'production',
  },
  media: {
    webrtc: MEDIAMTX_WEBRTC_URL,
    hls: MEDIAMTX_HLS_URL,
    rtsp: MEDIAMTX_RTSP_URL,
    mjpeg: BACKEND_STREAM_URL,
    // Use HLS as primary (from MediaMTX)
    primary: 'hls',
  },
  ui: {
    messageTimeout: 5000,
    animationsEnabled: true,
    sidebarCollapsible: true,
  },
  polling: {
    dashboard: 30000,
    violations: 30000,
    workers: 30000,
    alerts: 10000,
    cameras: 60000,
  },
};

// Logger utility
export const logger = {
  info: (...args) => config.logging.level !== 'silent' && console.log(...args),
  warn: (...args) => config.logging.level !== 'silent' && console.warn(...args),
  error: (...args) => console.error(...args),
  debug: (...args) => config.logging.enableDebug && console.debug(...args),
};

export default config;
