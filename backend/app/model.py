"""
Model inference wrapper with support for TensorFlow/Keras, PyTorch, TFLite, and Edge TPU.
Integrates with existing predict_frame function.
"""
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import numpy as np
import cv2

# Try TensorFlow first (for Keras models)
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    TF_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError as tf_error:
    TF_AVAILABLE = False
    print(f"❌ TensorFlow import failed: {tf_error}")

# Try PyTorch as fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("✅ PyTorch imported successfully")
except ImportError as torch_error:
    TORCH_AVAILABLE = False
    print(f"❌ PyTorch import failed: {torch_error}")

print(f"🔧 Final status - TF_AVAILABLE: {TF_AVAILABLE}, TORCH_AVAILABLE: {TORCH_AVAILABLE}")

from app.config import settings, SPECIES_CLASSES

logger = logging.getLogger(__name__)


class PlanktonModel:
    """
    Wrapper for plankton identification model.
    Supports hot-reloading, versioning, and rollback.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model_version = settings.MODEL_VERSION
        self.executor = ThreadPoolExecutor(max_workers=settings.INFERENCE_WORKERS)
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.species_classes = SPECIES_CLASSES

        # Temperature scaling for calibration (from your run_demo.py)
        self.temperature = getattr(settings, 'TEMPERATURE', 1.3132)
        self.img_size = getattr(settings, 'IMG_SIZE', 128)

        # Model history for rollback
        self.model_history = []

        # Initialize model-related attributes before loading
        self.model = None
        self.model_type = None
        self.device = getattr(settings, 'DEVICE', None)

        # Load initial model
        self.load_model()
    
    def update_temperature(self, temperature: float):
        """Update the temperature scaling parameter for calibration."""
        self.temperature = temperature
        logger.info(f"🔥 Temperature updated to: {temperature}")
        
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load model from disk. Supports hot-reloading for both Keras and PyTorch models.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = model_path or self.model_path
            path_obj = Path(path)
            
            logger.info(f"🔍 Attempting to load model from: {path}")
            logger.info(f"📁 Path exists: {path_obj.exists()}")
            logger.info(f"📄 Path is file: {path_obj.is_file()}")
            logger.info(f"🔧 TensorFlow available: {TF_AVAILABLE}")
            
            if not path_obj.exists():
                logger.warning(f"❌ Model file not found: {path}")
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Backup current model for rollback
            if self.model is not None:
                self.model_history.append({
                    "model": self.model,
                    "version": self.model_version,
                    "timestamp": time.time(),
                    "model_type": self.model_type
                })
                # Keep only last 3 versions
                if len(self.model_history) > 3:
                    self.model_history.pop(0)
            
            # Detect model type from file extension
            if path.endswith('.keras') and TF_AVAILABLE:
                logger.info("🎯 Detected Keras model, attempting to load using run_demo.py approach...")
                
                try:
                    # Use the exact same approach as run_demo.py - create fresh architecture and load weights
                    base_model = MobileNetV2(
                        input_shape=(self.img_size, self.img_size, 3),
                        include_top=False,
                        weights=None
                    )
                    inputs = Input(shape=(self.img_size, self.img_size, 3))
                    x = base_model(inputs, training=False)
                    x = GlobalAveragePooling2D()(x)
                    logits = Dense(len(self.species_classes), activation='linear')(x)
                    self.model = KerasModel(inputs, outputs=logits)
                    
                    # Load the trained weights (this is what works in run_demo.py)
                    self.model.load_weights(path)
                    self.model_type = 'keras'
                    logger.info(f"✅ Keras model loaded successfully using run_demo.py approach. Version: {self.model_version}")
                    
                except Exception as direct_load_error:
                    logger.warning(f"⚠️  run_demo.py approach failed: {direct_load_error}")
                    logger.info("🔄 Attempting alternative loading method...")
                    
                    try:
                        # Try loading with custom objects
                        from keras.src.models import Functional
                        custom_objects = {'Functional': Functional}
                        self.model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                        self.model_type = 'keras'
                        logger.info(f"✅ Keras model loaded with custom objects. Version: {self.model_version}")
                        
                    except Exception as custom_load_error:
                        logger.error(f"❌ All Keras loading methods failed. Model: {direct_load_error}, Custom: {custom_load_error}")
                        raise ValueError(f"Cannot load Keras model from {path}")
                    
            elif (path.endswith('.pth') or path.endswith('.pt')) and TORCH_AVAILABLE:
                # Load PyTorch model
                logger.info("🎯 Detected PyTorch model, attempting to load...")
                self.model_type = 'pytorch'
                checkpoint = torch.load(path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        self.model_version = checkpoint.get("version", self.model_version)
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Create model architecture
                self.model = self._create_model_architecture()
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"✅ PyTorch model loaded successfully. Version: {self.model_version}")
            else:
                logger.error("❌ Unsupported model format or missing dependencies")
                logger.error(f"   Path: {path}")
                logger.error(f"   Ends with .keras: {path.endswith('.keras')}")
                logger.error(f"   TF_AVAILABLE: {TF_AVAILABLE}")
                logger.error(f"   TORCH_AVAILABLE: {TORCH_AVAILABLE}")
                raise ValueError(f"Unsupported model format or missing dependencies for {path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"🔍 Full error traceback: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            
            logger.warning("⚠️  No model available. Creating dummy model.")
            self.model = self._create_dummy_model()
            return False
    
    def _create_model_architecture(self):
        """
        Create PyTorch model architecture. Only used for PyTorch models.
        This is a placeholder that matches common architectures.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Cannot create PyTorch model architecture.")
        
        # Example: ResNet-based classifier
        from torchvision.models import resnet18
        
        model = resnet18(pretrained=False)
        num_classes = len(self.species_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        return model
    
    def _create_dummy_model(self):
        """Create a dummy model for testing without a real model file."""
        if self.model_type == 'keras' and TF_AVAILABLE:
            # Create dummy Keras model
            inputs = Input(shape=(self.img_size, self.img_size, 3))
            x = GlobalAveragePooling2D()(inputs)
            logits = Dense(len(self.species_classes), activation='linear')(x)
            model = KerasModel(inputs, outputs=logits)
            logger.warning("🚨 USING DUMMY MODEL - THIS IS NOT YOUR TRAINED MODEL!")
            return model
        
        elif self.model_type == 'pytorch' and TORCH_AVAILABLE:
            # Create dummy PyTorch model
            class DummyModel(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    self.num_classes = num_classes
                
                def forward(self, x):
                    batch_size = x.shape[0]
                    # Return random logits
                    return torch.randn(batch_size, self.num_classes)
            
            model = DummyModel(len(self.species_classes))
            model.to(self.device)
            model.eval()
            logger.warning("🚨 USING DUMMY MODEL - THIS IS NOT YOUR TRAINED MODEL!")
            return model
        else:
            raise RuntimeError("No ML framework available (TensorFlow or PyTorch required)")
    
    def rollback(self) -> bool:
        """Rollback to previous model version."""
        if not self.model_history:
            logger.warning("No model history available for rollback")
            return False
        
        try:
            previous = self.model_history.pop()
            self.model = previous["model"]
            self.model_version = previous["version"]
            logger.info(f"Rolled back to model version {self.model_version}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def predict_frame_async(self, image: np.ndarray) -> Dict:
        """
        Async wrapper for predict_frame. Runs inference in thread pool.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Prediction dictionary with species, confidence, bbox, etc.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.predict_frame,
            image
        )
        return result
    
    def predict_frame(self, image: np.ndarray) -> Dict:
        """
        Main prediction function. Supports both Keras and PyTorch models.
        Implements temperature scaling calibration for Keras models.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            Dictionary containing:
                - species: str
                - confidence: float
                - bbox: [x1, y1, x2, y2] or None
                - timestamp: str (ISO format)
                - model_version: str
                - count: int
        """
        start_time = time.time()
        
        try:
            if self.model_type == 'keras' and TF_AVAILABLE:
                # Keras model prediction with temperature scaling (from your run_demo.py)
                # Preprocess image
                img_resized = cv2.resize(image, (self.img_size, self.img_size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_array = tf.expand_dims(img_rgb, 0)
                img_array_float = tf.cast(img_array, tf.float32)
                processed_array = preprocess_input(img_array_float)
                
                # DEBUG: Log preprocessing details
                logger.debug(f"📷 Image shape after resize: {img_resized.shape}")
                logger.debug(f"🎨 Image dtype after processing: {processed_array.dtype}")
                logger.debug(f"📊 Processed array shape: {processed_array.shape}")
                logger.debug(f"🔢 Processed array sample values: {processed_array[0, 0, 0, :5]}")
                
                # Get predictions from model
                predictions = self.model.predict(processed_array, verbose=0)
                
                # DEBUG: Log raw predictions before temperature scaling
                logger.debug(f"🔍 Raw logits: {predictions[0]}")
                logger.debug(f"🔥 Temperature: {self.temperature}")
                
                # Apply temperature scaling to logits
                scaled_logits = predictions / self.temperature
                
                # DEBUG: Log after temperature scaling
                logger.debug(f"📊 Scaled logits: {scaled_logits[0]}")
                
                # Apply softmax to get calibrated probabilities
                probs = tf.nn.softmax(scaled_logits[0])
                
                # DEBUG: Log probabilities before selecting max
                logger.debug(f"🎯 Probabilities: {probs.numpy()}")
                
                pred_class = int(np.argmax(probs))
                confidence = float(np.max(probs))
                
                # DEBUG: Log final results
                logger.debug(f"🏆 Final prediction: {self.species_classes[pred_class]} | Confidence: {confidence:.4f}")
                
                # Compare with run_demo.py logic
                run_demo_confidence = tf.nn.softmax(predictions[0]).numpy().max() * 100
                calibrated_confidence = confidence * 100
                
                logger.debug(f"📈 Comparison - Without temperature: {run_demo_confidence:.2f}% | With temperature: {calibrated_confidence:.2f}%")
                logger.debug(f"🔥 Temperature effect: {calibrated_confidence - run_demo_confidence:+.2f} percentage points")
                
            elif self.model_type == 'pytorch' and TORCH_AVAILABLE:
                # PyTorch model prediction
                input_tensor = self._preprocess_image(image)
                
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)
                    
                    confidence = confidence.item()
                    pred_class = pred_class.item()
            else:
                raise RuntimeError(f"No compatible model loaded. Model type: {self.model_type}")
            
            # Get species name
            species = self.species_classes[pred_class] if pred_class < len(self.species_classes) else "Unknown"
            
            # Optional: Detect bounding box (placeholder - implement your detection logic)
            bbox = self._detect_bbox(image, pred_class)
            
            # Update metrics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            result = {
                "species": species,
                "confidence": float(confidence),
                "bbox": bbox,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "model_version": self.model_version,
                "model_type": self.model_type,
                "count": 1 if confidence > settings.DEFAULT_CONFIDENCE_THRESHOLD else 0,
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            
            logger.debug(f"Prediction: {species} ({confidence:.2f}) in {inference_time*1000:.1f}ms")
            
            # Add detailed logging for debugging
            logger.info(f"🧬 PREDICTION: {species} | Confidence: {confidence:.3f} | Count: {1 if confidence > settings.DEFAULT_CONFIDENCE_THRESHOLD else 0} | Threshold: {settings.DEFAULT_CONFIDENCE_THRESHOLD}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                "species": "Error",
                "confidence": 0.0,
                "bbox": None,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "model_version": self.model_version,
                "count": 0,
                "error": str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for PyTorch model input.
        Only used for PyTorch models (Keras uses different preprocessing in predict_frame).
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor (1, C, H, W) for PyTorch
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for preprocessing")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (adjust as needed)
        image_resized = cv2.resize(image_rgb, (224, 224))
        
        # Normalize (ImageNet stats - adjust for your model)
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
        
        # Add batch dimension (1, C, H, W)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _detect_bbox(self, image: np.ndarray, pred_class: int) -> Optional[list]:
        """
        Detect bounding box for organism. Placeholder implementation.
        Replace with your actual detection logic (e.g., YOLO, Faster R-CNN).
        
        Args:
            image: Input image
            pred_class: Predicted class index
            
        Returns:
            [x1, y1, x2, y2] or None
        """
        # Placeholder: Return center region
        h, w = image.shape[:2]
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)
        
        # TODO: Implement actual detection logic
        # - Use object detection model (YOLO, Faster R-CNN)
        # - Or use segmentation + contour detection
        # - Or use attention maps from classifier
        
        return [x1, y1, x2, y2]
    
    def get_stats(self) -> Dict:
        """Get model performance statistics."""
        avg_time = (self.total_inference_time / self.inference_count * 1000 
                   if self.inference_count > 0 else 0)
        
        # Handle device string safely
        device_str = "CPU (TensorFlow)" if self.device is None else str(self.device)
        
        return {
            "model_version": self.model_version,
            "model_path": str(self.model_path),
            "device": device_str,
            "model_type": self.model_type,
            "inference_count": self.inference_count,
            "avg_inference_time_ms": round(avg_time, 2),
            "species_classes": self.species_classes,
            "num_classes": len(self.species_classes),
            "temperature": getattr(self, 'temperature', None),
            "img_size": getattr(self, 'img_size', None)
        }
    
    def save_model(self, path: str = None) -> bool:
        """Save current model to disk. Supports both Keras and PyTorch models."""
        try:
            save_path = path or self.model_path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            if self.model_type == 'keras' and TF_AVAILABLE:
                # Keras models are already saved as .keras files
                # For saving trained weights, we save the model directly
                if save_path.endswith('.keras'):
                    self.model.save(save_path)
                else:
                    # Save with .keras extension for compatibility
                    keras_path = save_path.rsplit('.', 1)[0] + '.keras'
                    self.model.save(keras_path)
                    logger.info(f"Keras model saved to {keras_path}")
                return True
            
            elif self.model_type == 'pytorch' and TORCH_AVAILABLE:
                # PyTorch checkpoint saving
                checkpoint = {
                    "model_state_dict": self.model.state_dict(),
                    "version": self.model_version,
                    "timestamp": time.time(),
                    "species_classes": self.species_classes
                }
                
                torch.save(checkpoint, save_path)
                logger.info(f"PyTorch model saved to {save_path}")
                return True
            else:
                raise ValueError(f"Cannot save model of type {self.model_type}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global model instance
model_instance: Optional[PlanktonModel] = None


def get_model() -> PlanktonModel:
    """Get or create global model instance."""
    global model_instance
    if model_instance is None:
        model_instance = PlanktonModel()
    return model_instance
