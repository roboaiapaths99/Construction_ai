// Centralized API service for AI detection
import config from '../config';

const API_BASE_URL = config.api.baseURL;

export async function detectObjects(imageBase64) {
  try {
    const response = await fetch(`${API_BASE_URL}/detect_base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        image: imageBase64 
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    
    if (result.error) {
      throw new Error(result.error);
    }

    return result;
  } catch (error) {
    console.error('Detection API error:', error);
    throw error;
  }
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Health check error:', error);
    return { status: 'unhealthy', model_loaded: false };
  }
}
