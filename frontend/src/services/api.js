const API_BASE = "http://localhost:8080";

export async function getCamera() {
  const res = await fetch(`${API_BASE}/cameras`);
  return res.json();
}

export async function getDetections(cameraId) {
  const res = await fetch(`${API_BASE}/cameras/${cameraId}/detections/latest`);
  return res.json();
}
