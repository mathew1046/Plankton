"""
WebSocket handler for real-time frame streaming and prediction.
"""
import asyncio
import logging
import time
from typing import Optional

import numpy as np
import cv2
from fastapi import WebSocket, WebSocketDisconnect, Query, status
from fastapi.websockets import WebSocketState

from app.model import get_model
from app.config import settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.connection_stats: dict[WebSocket, dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_stats[websocket] = {
            "connected_at": time.time(),
            "frames_received": 0,
            "frames_processed": 0,
            "errors": 0
        }
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            stats = self.connection_stats.pop(websocket, {})
            duration = time.time() - stats.get("connected_at", time.time())
            logger.info(
                f"WebSocket disconnected. Duration: {duration:.1f}s, "
                f"Frames: {stats.get('frames_processed', 0)}, "
                f"Errors: {stats.get('errors', 0)}"
            )
    
    async def send_json(self, websocket: WebSocket, data: dict):
        """Send JSON data to a specific connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(data)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "total_frames": sum(s.get("frames_received", 0) for s in self.connection_stats.values()),
            "total_processed": sum(s.get("frames_processed", 0) for s in self.connection_stats.values()),
            "total_errors": sum(s.get("errors", 0) for s in self.connection_stats.values())
        }


# Global connection manager
manager = ConnectionManager()


async def websocket_predict_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time frame prediction.
    
    Protocol:
    1. Client connects with token authentication
    2. Client sends binary JPEG frames
    3. Server decodes, runs inference, returns JSON prediction
    4. Repeat until client disconnects
    
    Args:
        websocket: FastAPI WebSocket connection
        token: Authentication token (API key)
    """
    # Authentication
    if token != settings.API_KEY:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        logger.warning(f"WebSocket auth failed. Token: {token}")
        return
    
    # Accept connection
    await manager.connect(websocket)
    model = get_model()
    
    try:
        # Send initial status
        await manager.send_json(websocket, {
            "type": "connected",
            "message": "WebSocket connection established",
            "model_version": model.model_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        })
        
        # Main message loop
        while True:
            try:
                # Receive frame data (binary JPEG)
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=settings.FRAME_TIMEOUT
                )
                
                manager.connection_stats[websocket]["frames_received"] += 1
                
                # Validate frame size
                if len(data) > settings.MAX_FRAME_SIZE:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": f"Frame too large: {len(data)} bytes"
                    })
                    continue
                
                # Decode JPEG
                frame = decode_frame(data)
                if frame is None:
                    await manager.send_json(websocket, {
                        "type": "error",
                        "message": "Failed to decode frame"
                    })
                    manager.connection_stats[websocket]["errors"] += 1
                    continue
                
                # Run inference asynchronously
                prediction = await model.predict_frame_async(frame)
                
                # Send prediction result
                prediction["type"] = "prediction"
                await manager.send_json(websocket, prediction)
                
                manager.connection_stats[websocket]["frames_processed"] += 1
                
            except asyncio.TimeoutError:
                # Send heartbeat if no frames received
                await manager.send_json(websocket, {
                    "type": "heartbeat",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                })
                
            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                break
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": str(e)
                })
                manager.connection_stats[websocket]["errors"] += 1
                
                # Disconnect on repeated errors
                if manager.connection_stats[websocket]["errors"] > 10:
                    logger.warning("Too many errors, closing connection")
                    break
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    
    finally:
        manager.disconnect(websocket)


def decode_frame(data: bytes) -> Optional[np.ndarray]:
    """
    Decode binary JPEG data to numpy array.
    
    Args:
        data: Binary JPEG data
        
    Returns:
        Decoded image as numpy array (H, W, C) in BGR format, or None if failed
    """
    try:
        # Decode JPEG from bytes
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("cv2.imdecode returned None")
            return None
        
        return frame
        
    except Exception as e:
        logger.error(f"Frame decode error: {e}")
        return None


def encode_frame(frame: np.ndarray, quality: int = 85) -> Optional[bytes]:
    """
    Encode numpy array to JPEG bytes.
    
    Args:
        frame: Image as numpy array (H, W, C) in BGR format
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG bytes or None if failed
    """
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encoded = cv2.imencode('.jpg', frame, encode_param)
        
        if not result:
            logger.error("cv2.imencode failed")
            return None
        
        return encoded.tobytes()
        
    except Exception as e:
        logger.error(f"Frame encode error: {e}")
        return None


async def broadcast_model_update(model_version: str):
    """Broadcast model update notification to all connected clients."""
    await manager.broadcast_json({
        "type": "model_update",
        "model_version": model_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": "Model has been updated. Predictions will use new version."
    })


def get_connection_stats() -> dict:
    """Get WebSocket connection statistics."""
    return manager.get_stats()
