const API_BASE = "";

export async function getCamera() {
  const res = await fetch(`${API_BASE}/cameras`);
  return res.json();
}

export async function getDetections(cameraId) {
  const res = await fetch(`${API_BASE}/cameras/${cameraId}/detections/latest`);
  return res.json();
}
