@echo off
REM FFmpeg bridge to stream IP camera to MediaMTX
REM Simplified version - direct passthrough without transcoding

echo Starting IP camera to MediaMTX bridge...
echo Camera: rtsp://192.168.1.16:554/11
echo Stream will be sent to MediaMTX path 'sitecam'
echo Press Ctrl+C to stop

"C:\Program Files\PictoBlox\resources\ffmpeg\win\ffmpeg.exe" -rtsp_transport udp -i rtsp://192.168.1.16:554/11 -c:v copy -an -f rtsp -rtsp_transport tcp rtsp://localhost:8554/sitecam
