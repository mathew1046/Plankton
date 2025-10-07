"""
Unit tests for WebSocket handler.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
import cv2
import numpy as np


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_frame_bytes():
    """Create sample frame as JPEG bytes."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()


def test_websocket_connection_no_token(client):
    """Test WebSocket connection without token."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/predict"):
            pass


def test_websocket_connection_invalid_token(client):
    """Test WebSocket connection with invalid token."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/predict?token=invalid"):
            pass


def test_websocket_connection_valid_token(client):
    """Test WebSocket connection with valid token."""
    with client.websocket_connect(f"/ws/predict?token={settings.API_KEY}") as websocket:
        # Should receive connection confirmation
        data = websocket.receive_json()
        assert data['type'] == 'connected'
        assert 'model_version' in data


def test_websocket_frame_prediction(client, sample_frame_bytes):
    """Test sending frame and receiving prediction."""
    with client.websocket_connect(f"/ws/predict?token={settings.API_KEY}") as websocket:
        # Receive connection confirmation
        websocket.receive_json()
        
        # Send frame
        websocket.send_bytes(sample_frame_bytes)
        
        # Receive prediction
        data = websocket.receive_json()
        
        assert data['type'] == 'prediction'
        assert 'species' in data
        assert 'confidence' in data
        assert 'timestamp' in data
        assert 'model_version' in data


def test_decode_frame(sample_frame_bytes):
    """Test frame decoding."""
    from app.websocket import decode_frame
    
    frame = decode_frame(sample_frame_bytes)
    
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3  # H, W, C


def test_encode_frame():
    """Test frame encoding."""
    from app.websocket import encode_frame
    
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    encoded = encode_frame(frame, quality=85)
    
    assert encoded is not None
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
