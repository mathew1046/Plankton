"""
Main FastAPI application for Marine Organism Identification System.
SIH 2025 - PS25043
"""
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings, SPECIES_CLASSES
from app.model import get_model
from app.websocket import websocket_predict_endpoint, get_connection_stats, broadcast_model_update
from app.training import get_training_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Real-time microscopic marine organism identification system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path(settings.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MODEL_BACKUP_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)


# Pydantic models for request/response
class TemperatureUpdateRequest(BaseModel):
    """Request model for temperature update."""
    temperature: float


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket, token: Optional[str] = Query(None)):
    """
    WebSocket endpoint for real-time frame prediction.

    Query Parameters:
        token: API authentication key

    Protocol:
        - Client sends: Binary JPEG frame data
        - Server responds: JSON prediction result
    """
    await websocket_predict_endpoint(websocket, token)


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/api/model/status")
async def get_model_status():
    """
    Get current model status and statistics.

    Returns:
        Model version, device, inference stats, and species classes
    """
    try:
        model = get_model()
        stats = model.get_stats()

        return JSONResponse({
            "status": "success",
            "data": stats,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        })
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/temperature")
async def get_temperature():
    """Get current model temperature setting."""
    try:
        model = get_model()
        current_temp = model.get_temperature()
        return JSONResponse({
            "status": "success",
            "temperature": current_temp
        })
    except Exception as e:
        logger.error(f"Failed to get temperature: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/temperature")
async def update_temperature(request: TemperatureUpdateRequest):
    """Update model temperature for confidence calibration."""
    try:
        if not 0.5 <= request.temperature <= 2.0:
            raise HTTPException(status_code=400, detail="Temperature must be between 0.5 and 2.0")

        model = get_model()
        model.update_temperature(request.temperature)

        return JSONResponse({
            "status": "success",
            "message": "Temperature updated",
            "temperature": request.temperature
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temperature update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/training/submit")
async def submit_label_correction(file: UploadFile = File(...),
                                  frame_id: str = Query(...),
                                  original_prediction: str = Query(...),
                                  corrected_label: str = Query(...),
                                  confidence: float = Query(...)):
    """
    Submit a corrected label for model training.

    Form Data:
        file: JPEG image file
        frame_id: Unique frame identifier
        original_prediction: Original model prediction
        corrected_label: User-corrected label
        confidence: Original prediction confidence

    Returns:
        Submission status and buffer statistics
    """
    try:
        # Validate corrected label
        if corrected_label not in SPECIES_CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid label. Must be one of: {SPECIES_CLASSES}"
            )

        # Read image data
        image_data = await file.read()

        # Submit to training manager
        training_mgr = get_training_manager()
        result = await training_mgr.submit_correction(
            image_data=image_data,
            original_prediction=original_prediction,
            corrected_label=corrected_label,
            confidence=confidence,
            frame_id=frame_id
        )

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/start")
async def start_training():
    """
    Start model fine-tuning on buffered samples.

    Returns:
        Training job status
    """
    try:
        training_mgr = get_training_manager()
        result = await training_mgr.start_training()

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/status")
async def get_training_status():
    """
    Get current training job status.

    Returns:
        Job status, progress, and metrics
    """
    try:
        training_mgr = get_training_manager()
        status = training_mgr.get_training_status()

        return JSONResponse({
            "status": "success",
            "data": status
        })

    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/buffer")
async def get_buffer_stats():
    """
    Get replay buffer statistics.

    Returns:
        Buffer size, utilization, and label distribution
    """
    try:
        training_mgr = get_training_manager()
        stats = training_mgr.get_buffer_stats()

        return JSONResponse({
            "status": "success",
            "data": stats
        })

    except Exception as e:
        logger.error(f"Failed to get buffer stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/training/buffer")
async def clear_buffer():
    """
    Clear the replay buffer.

    Returns:
        Success status
    """
    try:
        training_mgr = get_training_manager()
        training_mgr.clear_buffer()

        return JSONResponse({
            "status": "success",
            "message": "Buffer cleared"
        })

    except Exception as e:
        logger.error(f"Failed to clear buffer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.post("/api/snapshot")
async def save_snapshot(file: UploadFile = File(...),
                       frame_id: str = Query(...),
                       prediction: str = Query(...)):
    """
    Save a snapshot frame with prediction metadata.

    Form Data:
        file: JPEG image file
        frame_id: Unique frame identifier
        prediction: JSON string of prediction data

    Returns:
        Saved file path
    """
    try:
        import json

        # Parse prediction JSON
        pred_data = json.loads(prediction)

        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        species = pred_data.get("species", "unknown")
        filename = f"{timestamp}_{frame_id}_{species}.jpg"

        # Save image
        snapshot_path = Path(settings.SNAPSHOT_DIR) / filename
        image_data = await file.read()
        snapshot_path.write_bytes(image_data)

        # Save metadata
        metadata_path = snapshot_path.with_suffix('.json')
        metadata_path.write_text(json.dumps(pred_data, indent=2))

        logger.info(f"Snapshot saved: {filename}")

        return JSONResponse({
            "status": "success",
            "message": "Snapshot saved",
            "filename": filename,
            "path": str(snapshot_path)
        })

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service status and uptime
    """
    return JSONResponse({
        "status": "healthy",
        "service": "Marine Organism Identification API",
        "version": settings.API_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    })


@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics including WebSocket connections and model performance.

    Returns:
        Comprehensive system statistics
    """
    try:
        model = get_model()
        model_stats = model.get_stats()
        ws_stats = get_connection_stats()
        training_mgr = get_training_manager()
        buffer_stats = training_mgr.get_buffer_stats()

        return JSONResponse({
            "status": "success",
            "data": {
                "model": model_stats,
                "websocket": ws_stats,
                "training_buffer": buffer_stats
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        })

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/species")
async def get_species_list():
    """
    Get list of supported species classes.

    Returns:
        List of species names
    """
    return JSONResponse({
        "status": "success",
        "species": SPECIES_CLASSES,
        "count": len(SPECIES_CLASSES)
    })


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info("Marine Organism Identification System - Starting")
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"Model Path: {settings.MODEL_PATH}")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info("=" * 60)

    # Initialize model
    model = get_model()
    logger.info(f"Model loaded: {model.model_version}")

    # Initialize training manager
    training_mgr = get_training_manager()
    buffer_stats = training_mgr.get_buffer_stats()
    logger.info(f"Training buffer: {buffer_stats['size']} samples")

    logger.info("System ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")

    # Save training buffer
    training_mgr = get_training_manager()
    training_mgr.replay_buffer.save_to_disk()

    logger.info("Shutdown complete")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return JSONResponse({
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": "Real-time microscopic marine organism identification",
        "problem_statement": "SIH 2025 - PS25043",
        "endpoints": {
            "websocket": "/ws/predict?token=YOUR_API_KEY",
            "docs": "/docs",
            "health": "/api/health",
            "model_status": "/api/model/status",
            "temperature": "/api/model/temperature"
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
