#!/usr/bin/env python3
"""
Stream camera RTSP feed to MediaMTX
This script captures frames from the IP camera and publishes them to MediaMTX
"""

import cv2
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RTSP_INPUT = "rtsp://192.168.1.36"
RTSP_OUTPUT = "rtsp://localhost:8554/sitecam"
FFMPEG_CMD = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", RTSP_INPUT,
    "-an",  # Remove audio track to bypass codec issues
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_OUTPUT
]

def stream_to_mediamtx():
    """Stream camera to MediaMTX using ffmpeg"""
    logger.info(f"Starting camera stream from {RTSP_INPUT}")
    logger.info(f"Publishing to {RTSP_OUTPUT}")
    
    try:
        # Try with ffmpeg command line tool first
        process = subprocess.Popen(
            FFMPEG_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info("ffmpeg process started")
        
        # Keep the process running
        while True:
            if process.poll() is not None:
                logger.error(f"ffmpeg process exited with code {process.returncode}")
                time.sleep(5)  # Wait before restart
                break
            time.sleep(1)
            
    except FileNotFoundError:
        logger.error("ffmpeg not found! Trying alternative OpenCV method...")
        stream_with_opencv()
    except Exception as e:
        logger.error(f"Error: {e}")
        time.sleep(5)

def stream_with_opencv():
    """Fallback: Stream using OpenCV (slower, CPU intensive)"""
    logger.info("Using OpenCV for streaming (this is slower)")
    
    # For OpenCV streaming, we would need an RTSP server library like
    # mediamtx-python or similar. For now, just log that this is a fallback.
    logger.warning("OpenCV fallback requires additional RTSP server library")
    logger.warning("Please install ffmpeg for optimal performance")
    
    try:
        cap = cv2.VideoCapture(RTSP_INPUT)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
            
        logger.info("Camera opened successfully")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Cannot read frame from camera")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
            
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"OpenCV streaming error: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    logger.info("🎥 Camera to MediaMTX Bridge Starting...")
    
    # Try streaming in a loop with restart on failure
    while True:
        try:
            stream_to_mediamtx()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(5)
