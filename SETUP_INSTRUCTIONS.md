# 🚀 Setup Instructions - Marine Organism Identification System

## ⚡ Quick Setup (5 Minutes)

### Step 1: Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Start Backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Loading Keras model from ./models/plankton_classifier_seanoe_8_v2.keras
INFO:     Keras model loaded successfully. Version: 8.2
INFO:     System ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Frontend Setup (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

**Expected Output:**
```
VITE v5.0.8  ready in 500 ms
➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Step 4: Open Browser
Navigate to: **http://localhost:5173**

You should see a **stunning dark-themed UI** with:
- Animated gradient background
- Glassmorphic cards
- Glowing buttons
- Particle effects

---

## 🎨 What You'll See

### Visual Features
1. **Dark Space Theme** - Beautiful gradient background (slate → blue → purple)
2. **Glassmorphism** - Frosted glass cards with backdrop blur
3. **Animated Particles** - Floating glowing orbs in background
4. **Neon Buttons** - Cyan/blue gradient with glow effects
5. **Smooth Animations** - Cards float gently, buttons scale on hover
6. **Gradient Text** - Rainbow gradient on headings
7. **Live Predictions** - Real-time overlay with bounding boxes
8. **Species Counter** - Animated count display with badges

### Functional Features
1. **Camera Capture** - Access webcam/microscope
2. **Real-time AI** - Your calibrated model (T=1.3132)
3. **8 Species Detection** - All your trained classes
4. **Confidence Scores** - Properly calibrated probabilities
5. **FPS Control** - Adjust frame rate (1-10 FPS)
6. **Quality Control** - JPEG quality slider
7. **Threshold Filter** - Minimum confidence setting
8. **Label Correction** - Submit fixes for retraining
9. **Model Management** - Reload, rollback, version tracking
10. **Snapshot Capture** - Save frames with predictions

---

## 🧪 Testing

### Test 1: Check Backend Health
```bash
curl http://localhost:8000/api/health
```

**Expected:**
```json
{
  "status": "healthy",
  "service": "Marine Organism Identification API",
  "version": "1.0.0"
}
```

### Test 2: Check Model Status
```bash
curl http://localhost:8000/api/model/status
```

**Expected:**
```json
{
  "status": "success",
  "data": {
    "model_version": "8.2",
    "model_type": "keras",
    "device": "cpu",
    "species_classes": ["Appendicularia", "Calanoida", ...]
  }
}
```

### Test 3: Check Species List
```bash
curl http://localhost:8000/api/species
```

**Expected:**
```json
{
  "status": "success",
  "species": [
    "Appendicularia",
    "Calanoida",
    "Chaetognatha",
    "Diatoma",
    "Foraminifera",
    "Neoceratium",
    "Tomopteridae",
    "larvae_Echinodermata"
  ],
  "count": 8
}
```

### Test 4: WebSocket Stream Test
```bash
cd scripts
python test_stream.py --fps 6 --duration 10
```

**Expected:**
```
Connecting to ws://localhost:8000/ws/predict?token=dev-key-12345
✓ Connected to WebSocket
✓ Server response: WebSocket connection established
  Model version: 8.2
----------------------------------------
Streaming frames at 6 FPS for 10s...
Frame   1: Calanoida            (87.3%) - 45.2ms
Frame   2: Diatoma              (92.1%) - 43.8ms
...
```

---

## 🎯 Usage Workflow

### 1. Start Camera
1. Click **"Start"** button in Camera Feed card
2. Allow camera permissions when prompted
3. Video feed should appear

### 2. Begin Capture
1. Click **"Start Capture"** button (cyan gradient)
2. Watch for "🔴 Streaming frames to AI backend" status
3. Green connection indicator should show "Connected"

### 3. View Predictions
- **Bounding boxes** appear around detected organisms
- **Species name** displays with confidence percentage
- **Color coding**: Green (>80%), Yellow (60-80%), Orange (<60%)
- **Running count** updates in right panel

### 4. Adjust Settings
- **FPS Slider**: Control frame rate (lower = less CPU)
- **Confidence Threshold**: Filter low-confidence predictions
- **Quality Slider**: JPEG compression (lower = faster)

### 5. Correct Labels (Optional)
1. Click **"Correct Label"** in Label Correction card
2. Select correct species from dropdown
3. Click **"Submit"** to add to training buffer
4. Repeat until buffer has 10+ samples
5. Click **"Start Training"** in Model Status card

### 6. Save Snapshots
- Click **camera icon** next to "Stop Capture"
- Saves current frame + prediction to `backend/data/snapshots/`

---

## 🔧 Configuration

### Adjust Model Settings

Edit `backend/app/config.py`:
```python
MODEL_PATH = "./models/plankton_classifier_seanoe_8_v2.keras"
MODEL_VERSION = "8.2"
IMG_SIZE = 128
TEMPERATURE = 1.3132  # Your calibrated value
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
```

### Adjust UI Colors

Edit `frontend/tailwind.config.js`:
```javascript
colors: {
  primary: {
    500: '#06b6d4',  // Cyan
    600: '#3b82f6',  // Blue
  },
}
```

### Adjust Animations

Edit `frontend/src/index.css`:
```css
.float-animation {
  animation: float 6s ease-in-out infinite;  /* Adjust duration */
}
```

---

## 📊 Performance Tips

### For Best Performance:
- **FPS**: Start with 6 FPS, adjust based on CPU
- **Quality**: Use 80-85% for good balance
- **Threshold**: 0.5-0.6 filters noise
- **Workers**: 2-4 inference workers (backend config)

### If Slow:
- ✅ Reduce FPS to 4
- ✅ Lower quality to 70%
- ✅ Close other applications
- ✅ Use GPU if available (set `DEVICE=cuda` in backend)

### If Too Fast (wasting resources):
- ✅ Increase FPS to 8-10
- ✅ Increase quality to 90-95%
- ✅ Lower confidence threshold to 0.3

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error**: `ModuleNotFoundError: No module named 'tensorflow'`

**Fix**:
```bash
cd backend
venv\Scripts\activate
pip install tensorflow==2.15.0
```

### Model Not Found

**Error**: `Model file not found: ./models/plankton_classifier_seanoe_8_v2.keras`

**Fix**:
```bash
# Copy model from parent directory
copy ..\plankton_classifier_seanoe_8_v2.keras backend\models\
```

### Camera Not Working

**Error**: Camera permission denied or not detected

**Fix**:
- Use **Chrome** or **Edge** (best compatibility)
- Ensure you're on **localhost** or **HTTPS**
- Check browser settings → Privacy → Camera
- Try different camera if multiple available

### WebSocket Won't Connect

**Error**: Red "Disconnected" indicator

**Fix**:
1. Verify backend is running: `http://localhost:8000/api/health`
2. Check API key matches in frontend `.env` and backend config
3. Check browser console (F12) for errors
4. Restart both frontend and backend

### UI Not Loading Styles

**Error**: Plain white page, no styling

**Fix**:
```bash
cd frontend
npm install
npm run dev
```

### Predictions Too Slow

**Error**: Low FPS, laggy predictions

**Fix**:
- Reduce FPS to 4-6
- Lower JPEG quality to 70-75%
- Close other applications
- Check CPU usage (Task Manager)

---

## 📁 Project Structure

```
Dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py          ← FastAPI app
│   │   ├── model.py         ← Your Keras model integration
│   │   ├── config.py        ← Species classes + settings
│   │   └── websocket.py     ← Real-time streaming
│   ├── models/
│   │   └── plankton_classifier_seanoe_8_v2.keras  ← Your model
│   └── requirements.txt     ← TensorFlow + dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          ← Stunning UI (NEW!)
│   │   ├── index.css        ← Animations + glassmorphism
│   │   └── components/      ← React components
│   └── package.json
│
└── INTEGRATION_COMPLETE.md  ← Detailed integration guide
```

---

## 🎓 Key Concepts

### Temperature Scaling
Your model uses **T=1.3132** to calibrate confidence scores:
```python
scaled_logits = logits / temperature
calibrated_probs = softmax(scaled_logits)
```
This makes confidence scores more reliable!

### Glassmorphism
The UI uses frosted glass effects:
```css
background: rgba(255, 255, 255, 0.05);
backdrop-filter: blur(20px);
border: 1px solid rgba(255, 255, 255, 0.1);
```

### WebSocket Streaming
Binary JPEG frames sent via WebSocket for low latency:
```
Camera → Canvas → JPEG → WebSocket → Backend → Model → JSON → WebSocket → UI
```

---

## 🚀 Next Steps

### 1. Test with Real Data
- Connect microscope camera
- Test with your dataset images
- Verify predictions match expectations

### 2. Collect Training Data
- Use "Correct Label" feature
- Build replay buffer (10+ samples)
- Start fine-tuning to improve accuracy

### 3. Deploy to Production
- Use Docker: `docker-compose up --build`
- Deploy to cloud or edge device
- Set up monitoring and logging

### 4. Customize
- Adjust UI colors and animations
- Add custom species classes
- Implement additional features

---

## ✅ Success Checklist

- [ ] Backend starts without errors
- [ ] Model loads (check logs for "Keras model loaded successfully")
- [ ] Frontend shows beautiful dark UI with animations
- [ ] Camera access works
- [ ] WebSocket connects (green indicator)
- [ ] Predictions appear in real-time
- [ ] Species names are correct (your 8 classes)
- [ ] Confidence scores look reasonable (40-95%)
- [ ] Counts increment when organisms detected
- [ ] UI is smooth and responsive

---

## 🎉 You're Ready!

Your system is fully integrated and ready to use. Enjoy the stunning new interface with your calibrated plankton classifier!

**Need help?** Check:
- `INTEGRATION_COMPLETE.md` - Detailed integration guide
- `README.md` - Full documentation
- `http://localhost:8000/docs` - API documentation
- Backend logs: `backend/logs/`

---

**Built for SIH 2025 - PS25043**  
*Embedded Intelligent Microscopy System for Marine Organism Identification*
