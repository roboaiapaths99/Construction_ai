# Your IP Camera Configuration

## 🎥 Your Camera Details (From Code)

Based on your `rtsp_ai_camera.py` file, here's your setup:

```
Camera IP:     192.168.1.71
Port:          554
Stream:        /11
Protocol:      TCP (for stability)
Full URL:      rtsp://192.168.1.71:554/11?tcp
```

---

## ✅ TEST YOUR CAMERA NOW

### 3 Simple Commands to Run:

#### Command 1: Quick Check (Easiest)
```powershell
cd "C:\Users\Lenovo\Desktop\ai_construction_system - Copy"
.\.venv\Scripts\Activate.ps1
cd ai
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

**What this does:**
- Connects to your camera
- Tests for 60 seconds
- Shows detections in real-time
- Saves report with statistics

#### Command 2: Longer Test (Better Sample)
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 120
```

**What this does:**
- Same as above but 120 seconds
- Better statistics
- Captures more violations

#### Command 3: With Authentication (If Needed)
```powershell
python test_ip_camera.py "rtsp://admin:password@192.168.1.71:554/11?tcp" 60
```

**Use if:**
- Camera requires username/password
- Default is usually: admin/12345 or admin/admin

---

## 🔍 VERIFY BEFORE RUNNING

### Step 1: Camera is Online?
```powershell
ping 192.168.1.71
```

**Expected result:**
```
Reply from 192.168.1.71: bytes=32 time=10ms TTL=64
Reply from 192.168.1.71: bytes=32 time=11ms TTL=64
```

**If NOT working:**
- Camera is off
- Wrong IP address
- Network disconnected

### Step 2: Camera Is Accessible?
```powershell
# Try to reach it via HTTP
Start-Process "http://192.168.1.71"
```

**Expected result:**
- Camera login page opens
- Or shows camera interface

**If NOT working:**
- Check firewall
- Check camera password

---

## 🚀 RUN YOUR TEST

### Choose Your Command:

#### Option A: Standard Test (RECOMMENDED)
```powershell
# 60-second test on your camera
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

#### Option B: Extended Test (Better Data)
```powershell
# 120-second test on your camera
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 120
```

#### Option C: With Password
```powershell
# If camera needs login
python test_ip_camera.py "rtsp://admin:YOUR_PASSWORD@192.168.1.71:554/11?tcp" 60
```

---

## 📊 WHAT YOU'LL SEE

### Live Output:
```
📹 Connecting to IP Camera: rtsp://192.168.1.71:554/11?tcp
✅ Camera connected successfully!

🎬 Starting analysis...

Frame 10: Detections: 1 (no hat: 92%)
Frame 25: Detections: 2 (no vest: 85%, no boots: 81%)
Frame 45: Detections: 1 (no hat: 89%)

...more frames...

==== IP CAMERA TEST REPORT ====
Total Frames: 600
Total Detections: 45
Avg Detections/Frame: 0.075
Avg FPS: 10.3

Detected Classes:
  no hat:    15 detections, Avg 92%
  no vest:   18 detections, Avg 87%
  no boots:  12 detections, Avg 81%

Report saved: ip_camera_test_20260414_143022.json
```

---

## 🎮 DURING TEST - CONTROLS

While the test is running:

- **Press 'q'** → Stop test (anytime)
- **Press 's'** → Save current frame as image
- **ESC** → Also stops test

---

## 🔧 IF CONNECTION FAILS

### Error: "Failed to connect to camera"

**Try Fix 1: Remove ?tcp**
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11" 60
```

**Try Fix 2: Try with TCP explicitly**
```powershell
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

**Try Fix 3: Different stream path**
```powershell
# Try /stream
python test_ip_camera.py "rtsp://192.168.1.71:554/stream?tcp" 60

# Try /Streaming/Channels/101
python test_ip_camera.py "rtsp://192.168.1.71:554/Streaming/Channels/101?tcp" 60
```

**Try Fix 4: With authentication**
```powershell
python test_ip_camera.py "rtsp://admin:12345@192.168.1.71:554/11?tcp" 60
```

**Try Fix 5: Verify camera online**
```powershell
ping 192.168.1.71
```

If ping fails → Camera is offline or wrong IP

---

## 📋 CHECKLIST

Before running test:

- [ ] Virtual environment activated (`.\.venv\Scripts\Activate.ps1`)
- [ ] In `ai` folder (`cd ai`)
- [ ] Model exists (`models/yolov8n.pt`)
- [ ] Camera is powered on
- [ ] Camera IP is correct (192.168.1.71)
- [ ] Network is stable
- [ ] Have 60-120 seconds to spare

All checked? Run the command!

---

## 💡 PRO TIPS

1. **Position camera** to see people
2. **Move around in frame** during test
3. **Wear/don't wear PPE** to test detection
4. **Good lighting** helps detection
5. **Save image frames** of good detections (press 's')

---

## 📊 AFTER TEST - WHAT TO TELL ME

After the test completes, reply with:

```
Camera Test Results:
- Frames processed: [X]
- Detections: [Y]
- Violations detected: [list them]
- Any errors: [describe]
- Next step: [Path A/B/C?]
```

Example:
```
Camera Test Results:
- Frames processed: 600
- Detections: 45
- Violations detected: no hat (15x), no vest (18x), no boots (12x)
- Any errors: None
- Next step: Path A (go to production)
```

---

## 🚀 LET'S DO THIS!

### Your exact command to run:

```powershell
cd "C:\Users\Lenovo\Desktop\ai_construction_system - Copy"
.\.venv\Scripts\Activate.ps1
cd ai
python test_ip_camera.py "rtsp://192.168.1.71:554/11?tcp" 60
```

**Copy-paste this and run it now!**

---

## ✅ Expected Timeline

```
Activate environment:     30 seconds
Connect to camera:        5 seconds
Test running:             60 seconds
Analysis and report:      10 seconds
Total:                    ~90 seconds (1.5 min)
```

---

## 📞 NEXT STEPS

After test completes:

1. Review results (did it detect violations?)
2. Tell me what violations it detected
3. Choose your path (A = production, B = better model)
4. Continue to Phase 1 production readiness

---

**Ready? Run the command and let's see what your camera sees! 🎥🚀**
