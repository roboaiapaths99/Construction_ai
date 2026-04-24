import React from "react";

const CameraCard = ({ camera, onClick }) => {
  return (
    <div
      onClick={onClick}
      style={{
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "16px",
        cursor: "pointer",
        transition: "all 0.2s",
        backgroundColor: camera.status === "active" ? "#f0fff4" : "#fff5f5",
      }}
    >
      <h3 style={{ margin: "0 0 8px 0" }}>{camera.name}</h3>
      <div style={{ fontSize: "14px", color: "#666" }}>
        <p style={{ margin: "4px 0" }}>Status: {camera.status}</p>
        <p style={{ margin: "4px 0" }}>Location: {camera.location}</p>
        <p style={{ margin: "4px 0" }}>
          AI: {camera.ai_running ? "✅ Running" : "❌ Stopped"}
        </p>
      </div>
    </div>
  );
};

export default CameraCard;
