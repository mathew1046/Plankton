"""
Configuration management for the Marine Organism Identification System.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_KEY: str = "dev-key-12345"
    API_TITLE: str = "Marine Organism Identification API"
    API_VERSION: str = "1.0.0"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    # Model Configuration
    MODEL_PATH: str = "./models/plankton_classifier_seanoe_8_v2.keras"
    MODEL_VERSION: str = "8.2"
    MODEL_BACKUP_DIR: str = "./models/versions"
    
    # Model-specific settings (for Keras MobileNetV2)
    IMG_SIZE: int = 128
    TEMPERATURE: float = 1.3132  # Temperature scaling for calibration
    
    # Inference Configuration
    INFERENCE_WORKERS: int = 2
    MAX_FRAME_SIZE: int = 5 * 1024 * 1024  # 5MB
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds
    WS_MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Training Configuration
    TRAINING_BUFFER_SIZE: int = 1000
    TRAINING_BUFFER_PATH: str = "./data/replay_buffer"
    TRAINING_BATCH_SIZE: int = 16
    TRAINING_LEARNING_RATE: float = 1e-4
    TRAINING_EPOCHS: int = 10
    TRAINING_VALIDATION_SPLIT: float = 0.2
    TRAINING_MIN_ACCURACY: float = 0.85
    FREEZE_BACKBONE: bool = True  # Only train classifier head
    
    # Data Storage
    SNAPSHOT_DIR: str = "./data/snapshots"
    LOG_DIR: str = "./logs"
    
    # Device Configuration
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    USE_EDGE_TPU: bool = False  # Set to True for Coral Edge TPU
    
    # Performance
    MAX_QUEUE_SIZE: int = 100
    FRAME_TIMEOUT: int = 5  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Species mapping (from your trained model - run_demo.py)
SPECIES_CLASSES = [
    'Appendicularia',
    'Calanoida',
    'Chaetognatha',
    'Diatoma',
    'Foraminifera',
    'Neoceratium',
    'Tomopteridae',
    'larvae_Echinodermata'
]


# Training configuration
TRAINING_CONFIG = {
    "learning_rate": settings.TRAINING_LEARNING_RATE,
    "epochs": settings.TRAINING_EPOCHS,
    "batch_size": settings.TRAINING_BATCH_SIZE,
    "freeze_backbone": settings.FREEZE_BACKBONE,
    "validation_split": settings.TRAINING_VALIDATION_SPLIT,
    "early_stopping_patience": 3,
    "min_accuracy_threshold": settings.TRAINING_MIN_ACCURACY,
}
