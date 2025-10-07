# System Architecture

## Overview

The Marine Organism Identification System is a full-stack application designed for real-time microscopic organism detection and classification with on-device learning capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    React Frontend (Vite)                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ Camera       │  │ Prediction   │  │ Control Panel    │ │ │
│  │  │ Capture      │  │ Overlay      │  │ & Settings       │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ Count        │  │ Label        │  │ Model Status     │ │ │
│  │  │ Display      │  │ Correction   │  │ & Management     │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
                    WebSocket (Binary JPEG)
                    REST API (JSON)
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                        SERVER LAYER                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   FastAPI Backend                           │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │              WebSocket Handler                        │  │ │
│  │  │  - Connection management                              │  │ │
│  │  │  - Frame receive/decode                               │  │ │
│  │  │  - Async prediction dispatch                          │  │ │
│  │  │  - Result streaming                                   │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │              REST API Endpoints                       │  │ │
│  │  │  - Model management (reload, rollback, status)        │  │ │
│  │  │  - Training (submit, start, status)                   │  │ │
│  │  │  - Utilities (snapshot, health, stats)                │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE LAYER                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Model Inference Engine                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ ThreadPool   │→ │ Preprocessing│→ │ Model Forward    │ │ │
│  │  │ Executor     │  │ Pipeline     │  │ Pass (PyTorch)   │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ Postprocess  │→ │ BBox         │→ │ Result           │ │ │
│  │  │ & Softmax    │  │ Detection    │  │ Formatting       │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Semi-Supervised Learning System                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ Replay       │→ │ Dataset      │→ │ Fine-tuning      │ │ │
│  │  │ Buffer       │  │ Creation     │  │ (Classifier Head)│ │ │
│  │  │ (FIFO 1000)  │  │              │  │                  │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │ │
│  │  │ Validation   │→ │ Model        │→ │ Safe Swap        │ │ │
│  │  │ & Metrics    │  │ Checkpoint   │  │ with Rollback    │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Model checkpoints (.pth)                                 │ │
│  │  - Training samples (JPEG + metadata)                       │ │
│  │  - Snapshots (JPEG + predictions)                           │ │
│  │  - Logs (structured JSON logs)                              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend (React + Vite)

**Technology Stack:**
- React 18 with hooks
- Vite for fast development
- TailwindCSS for styling
- Lucide React for icons
- Native WebSocket API

**Key Components:**
1. **CameraCapture**: Manages camera access via `getUserMedia`, displays live feed
2. **PredictionOverlay**: Renders bounding boxes and labels over video feed
3. **ControlPanel**: FPS, confidence threshold, and quality controls
4. **CountDisplay**: Real-time organism counting by species
5. **LabelCorrection**: UI for submitting corrected labels
6. **ModelStatus**: Model info, training status, buffer stats

**Custom Hooks:**
- `useWebSocket`: WebSocket connection management with auto-reconnect
- `useCamera`: Camera device enumeration and stream management

### Backend (FastAPI)

**Technology Stack:**
- FastAPI for async HTTP/WebSocket
- Uvicorn ASGI server
- PyTorch for model inference
- OpenCV for image processing
- Pydantic for validation

**Key Modules:**
1. **main.py**: FastAPI app, REST endpoints, startup/shutdown
2. **websocket.py**: WebSocket handler, connection manager, frame decode
3. **model.py**: Model wrapper, inference, hot-reload, rollback
4. **training.py**: Replay buffer, fine-tuning, validation
5. **config.py**: Settings management with environment variables

**API Endpoints:**
- `WS /ws/predict`: Real-time frame streaming
- `GET /api/model/status`: Model information
- `POST /api/model/reload`: Hot reload model
- `POST /api/model/rollback`: Revert to previous version
- `POST /api/training/submit`: Submit corrected label
- `POST /api/training/start`: Start fine-tuning
- `GET /api/training/status`: Training job status
- `POST /api/snapshot`: Save frame with prediction

### Inference Pipeline

**Flow:**
1. Receive binary JPEG from WebSocket
2. Decode with OpenCV (`cv2.imdecode`)
3. Preprocess: resize, normalize, convert to tensor
4. Run inference in ThreadPoolExecutor (non-blocking)
5. Postprocess: softmax, argmax, bbox detection
6. Format result as JSON
7. Send back via WebSocket

**Performance:**
- Async WebSocket handling prevents blocking
- ThreadPool for CPU-bound inference
- Configurable worker count
- Frame queue with timeout

### Training System

**Semi-Supervised Learning:**
1. **Data Collection**: User corrects misclassifications via UI
2. **Replay Buffer**: FIFO buffer stores corrected samples (max 1000)
3. **Trigger**: Manual or automatic when buffer threshold reached
4. **Fine-tuning**: Only train classifier head (freeze backbone)
5. **Validation**: Split data, compute accuracy
6. **Safety**: Rollback if accuracy below threshold
7. **Versioning**: All models saved with metadata

**Training Configuration:**
- Learning rate: 1e-4
- Epochs: 10 (with early stopping)
- Batch size: 16
- Validation split: 20%
- Min accuracy: 85%

## Data Flow

### Prediction Flow
```
Camera → Canvas → JPEG Encode → WebSocket Send
                                      ↓
                              Backend Receive
                                      ↓
                              Decode + Preprocess
                                      ↓
                              Model Inference
                                      ↓
                              Postprocess + Format
                                      ↓
                              WebSocket Send
                                      ↓
UI Update ← JSON Parse ← WebSocket Receive
```

### Training Flow
```
User Correction → Form Submit → Backend API
                                      ↓
                              Replay Buffer Add
                                      ↓
                              Save to Disk
                                      ↓
                        (When triggered)
                                      ↓
                              Load Buffer Samples
                                      ↓
                              Create Dataset
                                      ↓
                              Fine-tune Classifier
                                      ↓
                              Validate Accuracy
                                      ↓
                        (If accuracy OK)
                                      ↓
                              Save Checkpoint
                                      ↓
                              Hot Reload Model
                                      ↓
                              Broadcast Update
```

## Deployment Options

### Development
- Frontend: Vite dev server (port 5173)
- Backend: Uvicorn with reload (port 8000)
- No containerization

### Docker Compose
- Frontend: Nginx container (port 3000)
- Backend: Python container (port 8000)
- Shared network
- Volume mounts for data persistence

### Edge Deployment

**Raspberry Pi 4:**
- TensorFlow Lite runtime
- INT8 quantized model
- 2-4 inference workers
- ~5-7 FPS

**Coral Edge TPU:**
- Edge TPU compiler
- Dedicated ML accelerator
- ~30-60 FPS
- Low power consumption

**Jetson Nano:**
- TensorRT optimization
- FP16 precision
- GPU acceleration
- ~15-25 FPS

## Security

**Authentication:**
- API key in WebSocket query parameter
- Token validation on connection
- Configurable via environment variable

**CORS:**
- Whitelist allowed origins
- Configurable for production

**Input Validation:**
- Max frame size limit (5MB)
- JPEG format validation
- Species label validation

## Performance Considerations

**Latency Optimization:**
- Binary WebSocket (not base64)
- Async I/O throughout
- ThreadPool for inference
- No unnecessary serialization

**Throughput:**
- Configurable FPS (1-10)
- JPEG quality control (50-100)
- Connection pooling
- Frame queue management

**Resource Usage:**
- CPU: 1-2 cores per inference worker
- Memory: 1-3GB depending on model
- Network: ~50-500 KB/s per stream
- Disk: Minimal (logs + snapshots)

## Scalability

**Horizontal Scaling:**
- Multiple backend instances behind load balancer
- Shared model storage (NFS/S3)
- Redis for session management
- Message queue for training jobs

**Vertical Scaling:**
- Increase inference workers
- Larger model capacity
- GPU acceleration
- Batch inference

## Monitoring

**Metrics:**
- Inference count and latency
- WebSocket connections
- Training buffer utilization
- Model version tracking

**Logging:**
- Structured JSON logs
- Per-request tracing
- Error tracking
- Performance profiling

## Future Enhancements

1. **Multi-model support**: Switch between models for different organism types
2. **Video recording**: Save full sessions with predictions
3. **Export**: CSV/Excel reports of counts and detections
4. **Alerts**: Notify on specific species detection
5. **Federated learning**: Aggregate models from multiple devices
6. **Mobile app**: iOS/Android clients
7. **Cloud sync**: Backup data and models to cloud storage
8. **Advanced analytics**: Time-series analysis, trends, patterns
