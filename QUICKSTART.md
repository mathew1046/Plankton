# Quick Start Guide

Get the Marine Organism Identification System running in 5 minutes!

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Camera** (webcam or microscope camera)
- **Docker** (optional, for containerized deployment)

## Option 1: Local Development (Fastest)

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at `http://localhost:8000`

### Step 2: Frontend Setup

Open a **new terminal**:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux

# Start the development server
npm run dev
```

Frontend will be available at `http://localhost:5173`

### Step 3: Access the Application

1. Open your browser to `http://localhost:5173`
2. Click **"Start"** to enable camera access
3. Click **"Start Capture"** to begin real-time predictions
4. Adjust settings as needed (FPS, confidence threshold)

## Option 2: Docker Compose (Production-like)

```bash
# From project root directory
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

To stop:
```bash
docker-compose down
```

## Testing the System

### Test WebSocket Connection

```bash
cd scripts
python test_stream.py --fps 6 --duration 30
```

This will:
- Connect to the WebSocket endpoint
- Stream synthetic frames for 30 seconds
- Display predictions in real-time
- Show performance statistics

### Load Testing

```bash
python load_test.py --connections 5 --fps 6 --duration 10
```

This simulates 5 concurrent connections for load testing.

## Troubleshooting

### Camera Not Working

**Issue**: Camera not detected or permission denied

**Solutions**:
- Ensure browser has camera permissions
- Use HTTPS or localhost (required for `getUserMedia`)
- Check if another application is using the camera
- Try a different browser (Chrome/Edge recommended)

### WebSocket Connection Failed

**Issue**: "WebSocket connection error" or "Disconnected"

**Solutions**:
- Verify backend is running (`http://localhost:8000/api/health`)
- Check API key matches in both frontend and backend `.env` files
- Ensure no firewall blocking port 8000
- Check browser console for detailed errors

### Backend Won't Start

**Issue**: Import errors or module not found

**Solutions**:
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+
```

### Model Not Found Warning

**Issue**: "Model file not found" warning in backend logs

**Solution**: This is expected on first run. The system creates a dummy model for testing. To use a real model:

1. Place your trained model at `backend/models/plankton_model.pth`
2. Or update `MODEL_PATH` in `backend/.env`
3. Restart the backend

### Low FPS / High Latency

**Solutions**:
- Reduce FPS in UI settings (try 4-6 FPS)
- Lower JPEG quality (try 70-80%)
- Increase `INFERENCE_WORKERS` in backend config
- Close other applications using CPU/GPU
- Consider using a quantized model for edge devices

## Next Steps

### 1. Customize Species Classes

Edit `backend/app/config.py`:

```python
SPECIES_CLASSES = [
    "YourSpecies1",
    "YourSpecies2",
    # ... add your species
]
```

### 2. Add Your Trained Model

Replace the dummy model:

```bash
# Copy your model to backend/models/
cp /path/to/your/model.pth backend/models/plankton_model.pth

# Restart backend
```

### 3. Configure Settings

Edit `.env` files in `backend/` and `frontend/`:

```bash
# Backend settings
API_KEY=your-secure-key-here
INFERENCE_WORKERS=4
DEVICE=cuda  # if you have GPU

# Frontend settings
VITE_API_KEY=your-secure-key-here
VITE_DEFAULT_FPS=8
```

### 4. Test Label Correction & Training

1. Make predictions on some frames
2. Click "Correct Label" if prediction is wrong
3. Select correct species and submit
4. Repeat until buffer has 10+ samples
5. Click "Start Training" in Model Status panel
6. Monitor training progress

### 5. Save Snapshots

Click the camera icon next to "Stop Capture" to save frames with predictions to `backend/data/snapshots/`

## API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Common Workflows

### Workflow 1: Real-time Detection

1. Start camera
2. Start capture
3. Observe predictions and counts
4. Adjust confidence threshold as needed
5. Save interesting snapshots

### Workflow 2: Model Improvement

1. Collect misclassified samples via label correction
2. Wait for buffer to fill (or manually trigger at 10+ samples)
3. Start training
4. Monitor validation accuracy
5. New model automatically loaded if accuracy is good
6. Continue detection with improved model

### Workflow 3: Batch Analysis

1. Record video of microscope feed
2. Use test script to stream video frames:
   ```bash
   python scripts/test_stream.py --image /path/to/frame.jpg --duration 60
   ```
3. Collect statistics from API:
   ```bash
   curl http://localhost:8000/api/stats
   ```

## Performance Tips

### For Raspberry Pi 4

```bash
# Use TFLite model (convert first)
DEVICE=cpu
INFERENCE_WORKERS=2
# Reduce FPS to 4-6
# Use JPEG quality 70-80
```

### For Desktop with GPU

```bash
# Use CUDA
DEVICE=cuda
INFERENCE_WORKERS=4
# Can use higher FPS (8-10)
# JPEG quality 85-95
```

### For Coral Edge TPU

```bash
# Enable Edge TPU
USE_EDGE_TPU=true
DEVICE=cpu  # CPU handles preprocessing
# Can achieve 30-60 FPS
```

## Getting Help

- Check `README.md` for detailed documentation
- Review `ARCHITECTURE.md` for system design
- Check backend logs: `backend/logs/`
- Check browser console for frontend errors
- API health check: `http://localhost:8000/api/health`

## What's Next?

- Explore the API documentation
- Customize the UI (edit React components)
- Train your own model
- Deploy to edge device
- Set up continuous monitoring

Happy detecting! 🔬🦠
