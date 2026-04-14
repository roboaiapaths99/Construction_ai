const navLinks = document.querySelectorAll(".nav-link");
const sections = document.querySelectorAll(".content-section");
const pageTitle = document.getElementById("page-title");

navLinks.forEach((link) => {
  link.addEventListener("click", function (e) {
    e.preventDefault();

    navLinks.forEach((item) => item.classList.remove("active"));
    this.classList.add("active");

    const target = this.getAttribute("data-section");

    sections.forEach((section) => section.classList.remove("active-section"));
    document.getElementById(target).classList.add("active-section");

    pageTitle.textContent = target.charAt(0).toUpperCase() + target.slice(1);
  });
});

const BASE_URL = "http://127.0.0.1:8000";

async function fetchData(endpoint) {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${endpoint}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    return [];
  }
}

function getBadgeClass(value) {
  const text = String(value).toLowerCase();

  if (text === "resolved" || text === "active" || text === "low") {
    return "badge-green";
  }
  if (text === "medium" || text === "on leave") {
    return "badge-yellow";
  }
  return "badge-red";
}

async function loadStats() {
  try {
    const response = await fetch(`${BASE_URL}/dashboard/stats`);
    const stats = await response.json();

    document.getElementById("total-workers").textContent = stats.total_workers ?? 0;
    document.getElementById("total-violations").textContent = stats.total_violations ?? 0;
    document.getElementById("total-alerts").textContent = stats.active_alerts ?? 0;
    document.getElementById("total-cameras").textContent = stats.connected_cameras ?? 0;
  } catch (error) {
    console.error("Error loading stats:", error);
  }
}

function loadViolations(violations) {
  const table = document.getElementById("violations-table");
  const recentViolations = document.getElementById("recent-violations");

  if (!violations.length) {
    table.innerHTML = `
      <tr>
        <td colspan="4" class="muted center">No violations loaded.</td>
      </tr>
    `;
    recentViolations.innerHTML = `<p class="muted">No data loaded yet.</p>`;
    return;
  }

  table.innerHTML = violations
    .map(
      (v) => `
      <tr>
        <td>${v.id}</td>
        <td>${v.worker}</td>
        <td>${v.type}</td>
        <td><span class="badge ${getBadgeClass(v.status)}">${v.status}</span></td>
      </tr>
    `
    )
    .join("");

  recentViolations.innerHTML = violations
    .slice(0, 3)
    .map(
      (v) => `
      <div class="list-item">
        <div>
          <strong>${v.worker}</strong><br>
          <span class="muted">${v.type}</span>
        </div>
        <span class="badge ${getBadgeClass(v.status)}">${v.status}</span>
      </div>
    `
    )
    .join("");
}

function loadWorkers(workers) {
  const grid = document.getElementById("workers-grid");

  if (!workers.length) {
    grid.innerHTML = `<p class="muted">No workers loaded.</p>`;
    return;
  }

  grid.innerHTML = workers
    .map(
      (w) => `
      <div class="worker-card">
        <h4>${w.name}</h4>
        <p><strong>Role:</strong> ${w.role}</p>
        <p><strong>Status:</strong> ${w.status}</p>
        <p><strong>ID:</strong> ${w.id}</p>
      </div>
    `
    )
    .join("");
}

function loadAlerts(alerts) {
  const list = document.getElementById("alerts-list");
  const recentAlerts = document.getElementById("recent-alerts");

  if (!alerts.length) {
    list.innerHTML = `<p class="muted">No alerts loaded.</p>`;
    recentAlerts.innerHTML = `<p class="muted">No data loaded yet.</p>`;
    return;
  }

  const html = alerts
    .map(
      (a) => `
      <div class="list-item">
        <div>${a.message}</div>
        <span class="badge ${getBadgeClass(a.level)}">${String(a.level).toUpperCase()}</span>
      </div>
    `
    )
    .join("");

  list.innerHTML = html;
  recentAlerts.innerHTML = html;
}

function loadCameras(cameras) {
  const cameraGrid = document.getElementById("camera-grid");

  if (!cameras.length) {
    cameraGrid.innerHTML = `<p class="muted">No cameras loaded.</p>`;
    return;
  }

  cameraGrid.innerHTML = cameras
    .map(
      (camera) => `
      <div class="camera-card">
        <div class="camera-screen">Camera ${camera.id}</div>
        <p>${camera.name}</p>
      </div>
    `
    )
    .join("");
}

async function initDashboard() {
  await loadStats();

  const [workers, violations, alerts, cameras] = await Promise.all([
    fetchData("/workers"),
    fetchData("/violations"),
    fetchData("/alerts"),
    fetchData("/cameras"),
  ]);

  loadWorkers(workers);
  loadViolations(violations);
  loadAlerts(alerts);
  loadCameras(cameras);
}

initDashboard();