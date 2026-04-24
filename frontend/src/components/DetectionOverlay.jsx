import React from "react";

const DetectionOverlay = ({ detections }) => {
  return (
    <div className="overlay">
      {detections.map((det, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: det.bbox.x,
            top: det.bbox.y,
            width: det.bbox.width,
            height: det.bbox.height,
            border: "2px solid red",
            color: "white",
            fontSize: "12px",
          }}
        >
          {det.class_name} ({det.confidence})
        </div>
      ))}
    </div>
  );
};

export default DetectionOverlay;
