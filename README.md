# Marine Organism Identification System - SIH 2025 PS25043

## Problem Statement Overview

**PS25043: Embedded Intelligent Microscopy System for Identification and Counting of Microscopic Marine Organisms**

This system provides real-time identification and counting of microscopic marine organisms (plankton, microalgae, etc.) using computer vision and deep learning. The solution enables:
- Live microscope camera feed capture via browser
- Real-time species identification with bounding boxes
- Automated counting and tracking
- On-device semi-supervised learning for model improvement
- Low-latency WebSocket streaming for immediate feedback

**Target Deployment:** Raspberry Pi 4, Jetson Nano, or systems with Coral Edge TPU for edge inference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser UI                            │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Camera       │→ │ Canvas      │→ │ WebSocket Client │   │
│  │ Capture      │  │ (JPEG enc.) │  │ (Binary frames)  │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
│         ↓                                     ↓               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Predictions Overlay (BBox, Species, Confidence)      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↕ WebSocket (wss://)
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ WS Handler   │→ │ Frame Queue │→ │ Model Inference  │   │
│  │ /ws/predict  │  │ (asyncio)   │  │ (ThreadPool)     │   │
│  └──────────────┘  └─────────────┘  └──────────────────┘   │
│         ↓                                     ↓               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ predict_frame(image) → {species, bbox, conf}         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  REST APIs: /api/model/reload, /api/model/status,           │
│             /api/snapshot, /api/training/submit              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                  ┌──────────────────┐
                  │ Self-Training    │
                  │ - Replay Buffer  │
                  │ - Fine-tune Head │
                  │ - Model Rollback │
                  └──────────────────┘
```

---

## Features

### Core Functionality
- ✅ **Real-time Camera Capture**: Browser-based microscope feed via `getUserMedia`
- ✅ **WebSocket Streaming**: Binary JPEG frames at 4-10 FPS
- ✅ **Live Predictions**: Species identification with confidence scores
- ✅ **Bounding Box Overlay**: Visual detection feedback
- ✅ **Running Count**: Track organism counts per species
- ✅ **Confidence Threshold**: Adjustable filter for predictions

### Advanced Features
- ✅ **Manual Label Correction**: UI for correcting misclassifications
- ✅ **Semi-Supervised Learning**: On-device classifier head retraining
- ✅ **Model Management**: Hot reload, version tracking, rollback
- ✅ **Snapshot Capture**: Save frames with predictions for dataset building
- ✅ **Logs & Monitoring**: Real-time inference logs and performance metrics

### Security & Production
- ✅ **Token Authentication**: API key for WebSocket connections
- ✅ **CORS Configuration**: Secure cross-origin requests
- ✅ **Docker Deployment**: Containerized frontend and backend
- ✅ **Edge TPU Ready**: Notes for Coral accelerator integration

---

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Docker & Docker Compose (optional)

### Local Development

#### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export API_KEY=your-secret-key-here
export MODEL_PATH=./models/plankton_model.pth

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Access the application at `http://localhost:5173`

### Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

## Configuration

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | WebSocket authentication key | `dev-key-12345` |
| `MODEL_PATH` | Path to trained model file | `./models/plankton_model.pth` |
| `MODEL_VERSION` | Current model version | `1.0.0` |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:5173"]` |
| `MAX_FRAME_SIZE` | Max frame size in bytes | `5242880` (5MB) |
| `INFERENCE_WORKERS` | ThreadPool size for inference | `2` |
| `TRAINING_BUFFER_SIZE` | Replay buffer size | `1000` |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_WS_URL` | WebSocket backend URL | `ws://localhost:8000` |
| `VITE_API_URL` | REST API backend URL | `http://localhost:8000` |
| `VITE_API_KEY` | API authentication key | `dev-key-12345` |
| `VITE_DEFAULT_FPS` | Default frame rate | `6` |

---

## API Documentation

### WebSocket Endpoint

**`/ws/predict?token={API_KEY}`**

- **Connect**: Establish WebSocket connection with token authentication
- **Send**: Binary JPEG frame data
- **Receive**: JSON prediction results

```json
{
  "species": "Copepod",
  "confidence": 0.94,
  "bbox": [120, 80, 340, 280],
  "timestamp": "2025-10-07T16:28:33.123Z",
  "model_version": "1.0.0",
  "count": 1
}
```

### REST Endpoints

#### Model Management

- **`GET /api/model/status`** - Get current model info
- **`POST /api/model/reload`** - Hot reload model from disk
- **`GET /api/model/versions`** - List available model versions
- **`POST /api/model/rollback`** - Rollback to previous version

#### Training & Data

- **`POST /api/training/submit`** - Submit corrected label for training
- **`POST /api/training/start`** - Start fine-tuning job
- **`GET /api/training/status`** - Get training job status

#### Utilities

- **`POST /api/snapshot`** - Save current frame with prediction
- **`GET /api/logs`** - Stream inference logs (SSE)
- **`GET /api/health`** - Health check

---

## Self-Training Workflow

The system supports on-device semi-supervised learning:

1. **Label Correction**: User corrects misclassified predictions via UI
2. **Replay Buffer**: Corrected samples stored locally (max 1000)
3. **Trigger Training**: Manual or automatic (when buffer reaches threshold)
4. **Fine-tune Head**: Only retrain classifier layer (fast, ~2-5 min)
5. **Validation**: Test new model on held-out samples
6. **Safe Swap**: Atomic model replacement with automatic rollback on failure
7. **Version Tracking**: All models versioned with metadata

### Training Configuration

```python
# backend/app/config.py
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 16,
    "freeze_backbone": True,  # Only train classifier head
    "validation_split": 0.2,
    "early_stopping_patience": 3,
    "min_accuracy_threshold": 0.85  # Rollback if below
}
```

---

## Edge Deployment

### Raspberry Pi 4 (4GB+)

**Recommended Setup:**
- TensorFlow Lite or ONNX Runtime
- Quantized INT8 model (~5-10 FPS)
- 2-4 inference workers

```bash
# Install TFLite runtime
pip install tflite-runtime

# Convert model to TFLite
python scripts/convert_to_tflite.py --model models/plankton_model.pth --quantize int8
```

### Coral Edge TPU

**Performance:** ~30-60 FPS with Edge TPU delegate

```bash
# Install Edge TPU runtime
bash scripts/install_edgetpu.sh

# Convert model to Edge TPU format
edgetpu_compiler models/plankton_model_quant.tflite
```

### Jetson Nano

**Recommended Setup:**
- TensorRT optimization
- FP16 precision (~15-25 FPS)

```bash
# Install TensorRT
sudo apt-get install tensorrt

# Convert model
python scripts/convert_to_tensorrt.py --model models/plankton_model.pth --precision fp16
```

### Resource Usage Estimates

| Platform | Inference Time | FPS | Memory | Power |
|----------|---------------|-----|--------|-------|
| Pi 4 (CPU) | ~150-200ms | 5-7 | 1.5GB | 5W |
| Pi 4 + Coral | ~15-30ms | 30-60 | 1.2GB | 7W |
| Jetson Nano | ~40-70ms | 15-25 | 2GB | 10W |
| Desktop GPU | ~5-10ms | 100+ | 3GB | 50W+ |

---

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests

```bash
cd frontend
npm run test
```

### Simulate Frame Stream

```bash
# Test WebSocket with sample frames
python scripts/test_stream.py --fps 10 --duration 30
```

### Load Testing

```bash
# Stress test WebSocket endpoint
python scripts/load_test.py --connections 10 --fps 10
```

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry
│   │   ├── websocket.py         # WebSocket handler
│   │   ├── model.py             # Model inference wrapper
│   │   ├── training.py          # Self-training logic
│   │   ├── config.py            # Configuration
│   │   └── utils.py             # Utilities
│   ├── models/
│   │   ├── plankton_model.pth   # Trained model (placeholder)
│   │   └── versions/            # Model version history
│   ├── data/
│   │   ├── replay_buffer/       # Training samples
│   │   └── snapshots/           # Saved frames
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app component
│   │   ├── components/
│   │   │   ├── CameraCapture.jsx
│   │   │   ├── PredictionOverlay.jsx
│   │   │   ├── ControlPanel.jsx
│   │   │   ├── CountDisplay.jsx
│   │   │   ├── LabelCorrection.jsx
│   │   │   └── ModelStatus.jsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js
│   │   │   └── useCamera.js
│   │   ├── utils/
│   │   │   └── frameProcessor.js
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── scripts/
│   ├── test_stream.py           # WebSocket test client
│   ├── load_test.py             # Load testing
│   ├── convert_to_tflite.py    # Model conversion
│   └── install_edgetpu.sh       # Edge TPU setup
│
├── docker-compose.yml           # Development compose
├── docker-compose.prod.yml      # Production compose
└── README.md
```

---

## Troubleshooting

### Camera Not Detected
- Ensure browser has camera permissions
- Use HTTPS or localhost (required for `getUserMedia`)
- Check browser console for errors

### WebSocket Connection Failed
- Verify API_KEY matches in frontend and backend
- Check CORS settings in backend config
- Ensure backend is running and accessible

### Low FPS / High Latency
- Reduce frame rate in UI (4-6 FPS recommended)
- Enable GPU acceleration if available
- Use quantized model for edge devices
- Increase `INFERENCE_WORKERS` for parallel processing

### Model Not Loading
- Verify `MODEL_PATH` points to valid model file
- Check model format matches inference code
- Review backend logs: `docker-compose logs backend`

---

## Contributing

This project is developed for SIH 2025. For issues or improvements:
1. Check existing issues
2. Create detailed bug reports with logs
3. Submit PRs with tests

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **SIH 2025** - Smart India Hackathon
- **Problem Statement**: PS25043 - Marine Organism Identification
- Built with FastAPI, React, TailwindCSS, and PyTorch
