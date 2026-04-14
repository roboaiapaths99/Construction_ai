import { apiEndpoints } from '../api';

class AIDetectionService {
  constructor() {
    this.isProcessing = false;
    this.detectionHistory = [];
    this.confidenceThreshold = 0.7;
  }

  // Process image for AI detection
  async detectObjects(imageData) {
    if (this.isProcessing) return null;
    
    this.isProcessing = true;
    
    try {
      // In a real implementation, you would send the image to your AI backend
      // For now, we'll simulate AI detection with mock data
      
      const mockDetection = await this.simulateAIDetection(imageData);
      
      // Store detection history
      this.detectionHistory.push({
        timestamp: new Date(),
        detections: mockDetection,
        imageData: imageData.substring(0, 100) + '...' // Store partial data for demo
      });

      // Keep only last 100 detections
      if (this.detectionHistory.length > 100) {
        this.detectionHistory = this.detectionHistory.slice(-100);
      }

      return mockDetection;
    } catch (error) {
      console.error('AI Detection Error:', error);
      return null;
    } finally {
      this.isProcessing = false;
    }
  }

  // Simulate AI detection (replace with real AI backend call)
  async simulateAIDetection(imageData) {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));

    // Mock detection results based on random factors
    const mockDetections = [];
    
    // Simulate person detection
    if (Math.random() > 0.3) {
      mockDetections.push({
        type: 'person',
        confidence: 0.8 + Math.random() * 0.2,
        bbox: [
          Math.floor(Math.random() * 200),
          Math.floor(Math.random() * 100),
          100 + Math.floor(Math.random() * 100),
          200 + Math.floor(Math.random() * 100)
        ],
        label: 'Worker'
      });
    }

    // Simulate safety equipment detection
    const equipmentTypes = [
      { type: 'hard_hat', label: 'Hard Hat', color: '#3b82f6' },
      { type: 'safety_vest', label: 'Safety Vest', color: '#f59e0b' },
      { type: 'safety_glasses', label: 'Safety Glasses', color: '#10b981' },
      { type: 'gloves', label: 'Gloves', color: '#8b5cf6' }
    ];

    equipmentTypes.forEach(equipment => {
      if (Math.random() > 0.6) {
        mockDetections.push({
          type: equipment.type,
          confidence: 0.7 + Math.random() * 0.3,
          bbox: [
            Math.floor(Math.random() * 300),
            Math.floor(Math.random() * 200),
            50 + Math.floor(Math.random() * 50),
            50 + Math.floor(Math.random() * 50)
          ],
          label: equipment.label,
          color: equipment.color
        });
      }
    });

    // Simulate violation detection
    const violations = [];
    
    // Check for missing safety equipment
    const hasPerson = mockDetections.some(d => d.type === 'person');
    const hasHardHat = mockDetections.some(d => d.type === 'hard_hat');
    const hasSafetyVest = mockDetections.some(d => d.type === 'safety_vest');

    if (hasPerson && !hasHardHat) {
      violations.push({
        type: 'no_hard_hat',
        severity: 'high',
        message: 'Worker without hard hat detected',
        confidence: 0.9
      });
    }

    if (hasPerson && !hasSafetyVest) {
      violations.push({
        type: 'no_safety_vest',
        severity: 'medium',
        message: 'Worker without safety vest detected',
        confidence: 0.8
      });
    }

    // Random safety zone violation
    if (Math.random() > 0.8) {
      violations.push({
        type: 'restricted_zone',
        severity: 'high',
        message: 'Person in restricted area',
        confidence: 0.85
      });
    }

    return {
      detections: mockDetections,
      violations: violations,
      timestamp: new Date(),
      processing_time: Math.floor(Math.random() * 500) + 100 // ms
    };
  }

  // Send violation to backend
  async reportViolation(violation, imageData) {
    try {
      const incident = {
        camera_name: 'Local_Webcam',
        violation_type: violation.type,
        persons: 1,
        timestamp: new Date().toISOString(),
        image_path: `webcam_${Date.now()}.jpg`,
        status: 'open',
        camera_ip: 'Local_Webcam'
      };

      // Send to backend
      const response = await apiEndpoints.createIncident(incident);
      
      console.log('Violation reported:', response.data);
      return response.data;
    } catch (error) {
      console.error('Failed to report violation:', error);
      throw error;
    }
  }

  // Get detection statistics
  getDetectionStats() {
    const lastHour = new Date(Date.now() - 60 * 60 * 1000);
    const recentDetections = this.detectionHistory.filter(d => d.timestamp > lastHour);

    const stats = {
      totalDetections: recentDetections.length,
      violations: recentDetections.reduce((acc, d) => acc + d.detections.violations.length, 0),
      personsDetected: recentDetections.reduce((acc, d) => 
        acc + d.detections.detections.filter(det => det.type === 'person').length, 0
      ),
      avgConfidence: recentDetections.reduce((acc, d) => {
        const confidences = d.detections.detections.map(det => det.confidence);
        return acc + (confidences.reduce((a, b) => a + b, 0) / confidences.length || 0);
      }, 0) / (recentDetections.length || 1)
    };

    return stats;
  }

  // Clear detection history
  clearHistory() {
    this.detectionHistory = [];
  }

  // Set confidence threshold
  setConfidenceThreshold(threshold) {
    this.confidenceThreshold = Math.max(0, Math.min(1, threshold));
  }
}

// Create singleton instance
const aiDetectionService = new AIDetectionService();

export default aiDetectionService;
