# IP Camera AI Model Test - Quick Start

## 🎥 Test Your AI Model on IP Camera

This is better than webcam testing because it shows real-world performance on actual construction footage!

---

## ⚡ QUICK COMMANDS

### Option 1: Interactive (Asks for URL)
```powershell
cd "C:\Users\Lenovo\Desktop\ai_construction_system - Copy"
.\.venv\Scripts\Activate.ps1
cd ai
python test_ip_camera.py
```

### Option 2: With URL (Auto-connect)
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

### Option 3: Longer test (2 minutes)
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 120
```

---

## 📍 Camera Configuration

### What you'll need:
1. **Camera IP Address**: 192.168.1.71 (adjust to yours)
2. **RTSP Port**: Usually 554
3. **Stream Path**: /11 or /stream or /Streaming/Channels/101
4. **Protocol**: RTSP with TCP for stability

### Common RTSP URL Formats:

**Generic IP Camera:**
```
rtsp://192.168.1.71:554/stream
rtsp://192.168.1.71:554/stream?tcp
```

**With Username/Password:**
```
rtsp://admin:12345@192.168.1.71:554/stream
```

**Hikvision Camera:**
```
rtsp://admin:password@192.168.1.71:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.71:554/Streaming/Channels/101?tcp
```

**Dahua Camera:**
```
rtsp://admin:password@192.168.1.71:554/stream
```

**Axis Camera:**
```
rtsp://192.168.1.71:554/axis-media/media.amp
```

---

## 🔍 Find Your Camera URL

### Step 1: Check Your Camera Documentation
- Look for RTSP URL in camera manual
- Usually under "Network" or "Streaming" settings

### Step 2: Test Connectivity
```powershell
# Test if camera is reachable
ping 192.168.1.71

# If successful, you'll see:
# Reply from 192.168.1.71: bytes=32 time=XX ms
```

### Step 3: Try Common URLs
```powershell
# Try these in order:
rtsp://192.168.1.71:554/stream
rtsp://192.168.1.71:554/11
rtsp://192.168.1.71:554/Streaming/Channels/101
rtsp://admin:12345@192.168.1.71:554/stream
```

### Step 4: Test with VLC (Optional)
```
1. Open VLC Media Player
2. File → Open Network Stream
3. Enter your RTSP URL
4. If it plays, that's your correct URL!
```

---

## ⚡ WHAT HAPPENS WHEN YOU RUN IT

### 1. Connection Phase
```
📹 Connecting to IP Camera: rtsp://192.168.1.71:554/11?tcp
✅ Camera connected successfully!
```

### 2. Analysis Phase
```
🎬 Starting analysis...
Press 'q' to quit early
Press 's' to save current frame
---

Frame 5: "no hat: 92%"
Frame 12: "no vest: 85%"
Frame 23: "no hat: 94%, no vest: 89%"
```

### 3. Results Phase
```
==== IP CAMERA TEST REPORT ====
Total Frames: 600
Total Detections: 45
Avg Detections/Frame: 0.075
Avg FPS: 10.3

Detected Classes:
  no hat: 15 detections, Avg 92%, Range 88%-95%
  no vest: 18 detections, Avg 87%, Range 81%-93%
  no boots: 12 detections, Avg 81%, Range 75%-89%

Report saved: ip_camera_test_20260414_143022.json
```

---

## 🎮 CONTROLS

While test is running:

- **'q'** → Quit test (stops analysis)
- **'s'** → Save current frame (as .jpg)
- **ESC** → Also quit

---

## 📊 WHAT IT TESTS

✅ **Real-world construction footage**
✅ **Multiple angles** (people moving in frame)
✅ **Lighting conditions** (whatever your camera sees)
✅ **Frame rate** (FPS measurement)
✅ **Accuracy** (detection confidence)

---

## ✨ EXPECTED RESULTS

### Good Results:
```
Total Frames: 300+
Total Detections: 20+
Avg FPS: 5-15 (depends on model)
Avg Confidence: 80%+
```

### Problems:
```
No detections → Check camera view angle
Low FPS → Camera/network is slow
Low confidence → Adjust camera angle/lighting
Connection failed → Check URL/camera online
```

---

## 🔧 IF CONNECTION FAILS

### Error: "Failed to connect to camera"

**Fix 1: Check camera is online**
```powershell
ping 192.168.1.71
```

**Fix 2: Try with ?tcp**
```
rtsp://192.168.1.71:554/11?tcp
```

**Fix 3: Add username/password**
```
rtsp://admin:12345@192.168.1.71:554/11?tcp
```

**Fix 4: Try different port**
```
rtsp://192.168.1.71:555/stream  (try 555)
rtsp://192.168.1.71:554/stream  (try 554)
```

**Fix 5: Check firewall**
```powershell
# Windows Firewall - allow port 554
netsh advfirewall firewall add rule name="RTSP" dir=in action=allow protocol=tcp localport=554
```

---

## 📝 QUICK START STEPS

### Step 1: Find Your Camera URL (2 min)
```
1. Check camera documentation
2. Or test with VLC
3. Note down the RTSP URL
```

### Step 2: Activate Environment (1 min)
```powershell
cd "C:\Users\Lenovo\Desktop\ai_construction_system - Copy"
.\.venv\Scripts\Activate.ps1
cd ai
```

### Step 3: Run Test (5 min)
```powershell
python test_ip_camera.py "YOUR_RTSP_URL_HERE" 60
```

### Step 4: Check Results (2 min)
```
Look for:
✅ Detections in console
✅ Test report saved
✅ Performance metrics
```

---

## 🎯 EXAMPLE RUNS

### Example 1: Simple URL
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

### Example 2: With Authentication
```powershell
python test_ip_camera.py "rtsp://admin:password@192.168.1.71:554/Streaming/Channels/101" 60
```

### Example 3: Interactive (Script Asks)
```powershell
python test_ip_camera.py
# Then type: rtsp://192.168.1.71:554/11?tcp
# Press Enter
```

---

## 📊 OUTPUT FILES

After test completes, you'll get:

### 1. Console Report
```
Shows real-time detections and summary
```

### 2. JSON Report File
```
ip_camera_test_20260414_143022.json

Contains:
- Frame count
- Detection statistics
- FPS measurements
- Detailed detection list
```

### 3. Saved Frames (if you pressed 's')
```
camera_frame_0.jpg
camera_frame_50.jpg
etc...
```

---

## 💡 TIPS FOR BEST RESULTS

1. **Position camera** to capture multiple people
2. **Test during activity** (people moving, working)
3. **Check lighting** (camera can see clearly)
4. **Run test for 60-120 seconds** for good sample
5. **Save interesting frames** (press 's')

---

## 🎯 WHAT TO DO NEXT

### After getting results:

1. **Did it detect violations?** ✅ Great! Continue to Phase 1
2. **No detections?** Check:
   - Camera angle
   - People wearing safety gear
   - Adjust confidence threshold (test_ip_camera.py line 70)
3. **Connection failed?** Try different URL

---

## 📞 EXAMPLE CAMERA CONFIGURATIONS

### Your Exact Camera (if Hikvision)
```powershell
python test_ip_camera.py "rtsp://admin:12345@192.168.1.71:554/Streaming/Channels/101?tcp" 60
```

### Your Exact Camera (if Generic)
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

Try these first!

---

## ⏱️ TIMING

```
Find URL:           5 min
Setup:              2 min
Run test:           1-2 min (test runs)
Review results:     3 min
Total:              15 min
```

---

## 🚀 READY?

1. Get your camera RTSP URL
2. Run the test command
3. Watch for detections
4. Tell me the results!

---

**Questions?** Check the troubleshooting section above.
**Ready to test?** Run the command now! 👇
