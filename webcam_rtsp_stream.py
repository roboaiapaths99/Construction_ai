#!/usr/bin/env python3
"""
Capture from laptop webcam and stream to MediaMTX via RTSP.
This allows testing the attendance system with your webcam.
"""

import cv2
import subprocess
import sys
import time

def stream_webcam_to_rtsp():
    """Capture from webcam and stream to MediaMTX RTSP server."""
    
    # Open webcam (0 is default laptop webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return False
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"Webcam opened: {width}x{height} @ {fps} fps")
    
    # FFmpeg command to receive raw video and stream via RTSP
    ffmpeg_path = r'C:\Program Files\PictoBlox\resources\ffmpeg\win\ffmpeg.exe'
    ffmpeg_cmd = [
        ffmpeg_path,
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height}',
        '-framerate', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-an',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        'rtsp://127.0.0.1:8554/sitecam'
    ]
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        print("FFmpeg process started, streaming to rtsp://127.0.0.1:8554/sitecam")
    except Exception as e:
        print(f"ERROR: Could not start FFmpeg: {e}")
        cap.release()
        return False
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from webcam")
                break
            
            # Send frame to FFmpeg
            process.stdin.write(frame.tobytes())
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Streaming... ({frame_count} frames)")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        cap.release()
        process.stdin.close()
        process.terminate()
        process.wait()
        print("Webcam stream stopped")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Laptop Webcam to RTSP Streamer")
    print("=" * 60)
    print("\nStarting webcam capture and streaming to MediaMTX...")
    print("Stream URL: rtsp://127.0.0.1:8554/sitecam")
    print("\nPress Ctrl+C to stop streaming\n")
    
    stream_webcam_to_rtsp()
