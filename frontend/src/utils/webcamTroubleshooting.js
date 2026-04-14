// Webcam Troubleshooting Utility

export const webcamTroubleshooting = {
  // Check browser compatibility
  checkBrowserCompatibility() {
    const userAgent = navigator.userAgent;
    const isChrome = /Chrome/.test(userAgent);
    const isFirefox = /Firefox/.test(userAgent);
    const isSafari = /Safari/.test(userAgent) && !/Chrome/.test(userAgent);
    const isEdge = /Edg/.test(userAgent);

    const compatibility = {
      supported: isChrome || isFirefox || isSafari || isEdge,
      browser: isChrome ? 'Chrome' : isFirefox ? 'Firefox' : isSafari ? 'Safari' : 'Edge',
      version: this.getBrowserVersion(userAgent),
      recommendations: []
    };

    if (!compatibility.supported) {
      compatibility.recommendations.push('Please use a modern browser like Chrome, Firefox, Safari, or Edge');
    }

    if (isSafari && parseInt(compatibility.version) < 11) {
      compatibility.recommendations.push('Please update Safari to version 11 or higher');
    }

    return compatibility;
  },

  getBrowserVersion(userAgent) {
    const match = userAgent.match(/(Chrome|Firefox|Safari|Edg)\/(\d+)/);
    return match ? match[2] : 'Unknown';
  },

  // Test camera access
  async testCameraAccess() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false 
      });
      
      // Check if we actually get video
      const videoTrack = stream.getVideoTracks()[0];
      const capabilities = videoTrack.getCapabilities();
      
      stream.getTracks().forEach(track => track.stop());
      
      return {
        success: true,
        capabilities: {
          width: capabilities.width?.max || 1280,
          height: capabilities.height?.max || 720,
          facingMode: capabilities.facingMode || ['user'],
          deviceId: capabilities.deviceId || 'default'
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.name,
        message: this.getErrorMessage(error)
      };
    }
  },

  getErrorMessage(error) {
    switch (error.name) {
      case 'NotAllowedError':
        return 'Camera permission denied. Please allow camera access in your browser settings.';
      case 'NotFoundError':
        return 'No camera found. Please connect a camera to your device.';
      case 'NotReadableError':
        return 'Camera is already in use by another application.';
      case 'OverconstrainedError':
        return 'Camera does not support the requested constraints.';
      case 'SecurityError':
        return 'Camera access blocked due to security restrictions.';
      case 'TypeError':
        return 'Camera API not supported in this browser.';
      default:
        return `Camera error: ${error.message}`;
    }
  },

  // Get troubleshooting steps based on error
  getTroubleshootingSteps(error) {
    const steps = [];

    switch (error) {
      case 'NotAllowedError':
        steps.push(
          '1. Click the camera icon 📷 in your browser\'s address bar',
          '2. Select "Allow" for camera access',
          '3. Refresh the page and try again',
          '4. Check if another tab is using the camera'
        );
        break;
      case 'NotFoundError':
        steps.push(
          '1. Check if your webcam is properly connected',
          '2. Try unplugging and reconnecting the camera',
          '3. Check if camera works in other applications',
          '4. Restart your computer if the issue persists'
        );
        break;
      case 'NotReadableError':
        steps.push(
          '1. Close other applications that might be using the camera',
          '2. Check if video conferencing apps are running',
          '3. Close browser tabs that might be using the camera',
          '4. Restart your browser'
        );
        break;
      case 'OverconstrainedError':
        steps.push(
          '1. Try using lower resolution settings',
          '2. Check if your camera supports the requested resolution',
          '3. Close other applications that might be using the camera'
        );
        break;
      default:
        steps.push(
          '1. Refresh the page and try again',
          '2. Clear your browser cache and cookies',
          '3. Try using a different browser',
          '4. Restart your computer'
        );
    }

    return steps;
  },

  // Check if HTTPS is required
  checkHTTPSRequirement() {
    const isHTTPS = location.protocol === 'https:';
    const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    
    return {
      required: !isHTTPS && !isLocalhost,
      currentProtocol: location.protocol,
      recommendation: isLocalhost 
        ? 'Localhost access is allowed'
        : 'Please use HTTPS or localhost for camera access'
    };
  },

  // Generate diagnostic report
  async generateDiagnosticReport() {
    const compatibility = this.checkBrowserCompatibility();
    const cameraTest = await this.testCameraAccess();
    const httpsCheck = this.checkHTTPSRequirement();

    return {
      timestamp: new Date().toISOString(),
      browser: compatibility,
      camera: cameraTest,
      security: httpsCheck,
      system: {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        language: navigator.language,
        cookieEnabled: navigator.cookieEnabled,
        onLine: navigator.onLine
      }
    };
  }
};

export default webcamTroubleshooting;
