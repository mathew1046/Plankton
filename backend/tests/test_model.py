"""
Unit tests for model inference module.
"""
import pytest
import numpy as np
import cv2
from app.model import PlanktonModel


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return PlanktonModel()


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


def test_model_initialization(model):
    """Test model initializes correctly."""
    assert model is not None
    assert model.model is not None
    assert model.inference_count == 0


def test_predict_frame(model, sample_frame):
    """Test frame prediction."""
    result = model.predict_frame(sample_frame)
    
    assert isinstance(result, dict)
    assert 'species' in result
    assert 'confidence' in result
    assert 'timestamp' in result
    assert 'model_version' in result
    assert 'count' in result
    
    assert isinstance(result['species'], str)
    assert 0 <= result['confidence'] <= 1
    assert isinstance(result['count'], int)


@pytest.mark.asyncio
async def test_predict_frame_async(model, sample_frame):
    """Test async frame prediction."""
    result = await model.predict_frame_async(sample_frame)
    
    assert isinstance(result, dict)
    assert 'species' in result


def test_model_stats(model, sample_frame):
    """Test model statistics."""
    # Make a prediction
    model.predict_frame(sample_frame)
    
    stats = model.get_stats()
    
    assert isinstance(stats, dict)
    assert stats['inference_count'] == 1
    assert 'avg_inference_time_ms' in stats
    assert 'model_version' in stats
    assert 'device' in stats


def test_preprocess_image(model, sample_frame):
    """Test image preprocessing."""
    tensor = model._preprocess_image(sample_frame)
    
    assert tensor.shape[0] == 1  # Batch size
    assert tensor.shape[1] == 3  # Channels
    assert tensor.shape[2] == 224  # Height
    assert tensor.shape[3] == 224  # Width


def test_model_reload(model):
    """Test model reload functionality."""
    initial_version = model.model_version
    success = model.load_model()
    
    assert success is True
    # Model should still be loaded even if file doesn't exist (dummy model)
    assert model.model is not None


def test_model_save(model, tmp_path):
    """Test model saving."""
    save_path = tmp_path / "test_model.pth"
    success = model.save_model(str(save_path))
    
    assert success is True
    assert save_path.exists()


def test_multiple_predictions(model, sample_frame):
    """Test multiple consecutive predictions."""
    results = []
    
    for _ in range(5):
        result = model.predict_frame(sample_frame)
        results.append(result)
    
    assert len(results) == 5
    assert model.inference_count == 5
    
    # All results should have required fields
    for result in results:
        assert 'species' in result
        assert 'confidence' in result
