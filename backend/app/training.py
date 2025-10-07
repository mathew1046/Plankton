"""
Semi-supervised self-training module for on-device model improvement.
"""
import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from app.config import settings, TRAINING_CONFIG, SPECIES_CLASSES
from app.model import get_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Represents a single training sample with corrected label."""
    image_data: bytes  # JPEG bytes
    original_prediction: str
    corrected_label: str
    confidence: float
    timestamp: float
    frame_id: str


class ReplayBuffer:
    """
    Circular buffer for storing corrected training samples.
    Implements FIFO eviction when buffer is full.
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.TRAINING_BUFFER_SIZE
        self.buffer: deque[TrainingSample] = deque(maxlen=self.max_size)
        self.buffer_path = Path(settings.TRAINING_BUFFER_PATH)
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing buffer from disk
        self.load_from_disk()
    
    def add(self, sample: TrainingSample):
        """Add a sample to the buffer."""
        self.buffer.append(sample)
        logger.info(f"Added sample to buffer. Size: {len(self.buffer)}/{self.max_size}")
    
    def get_all(self) -> List[TrainingSample]:
        """Get all samples in the buffer."""
        return list(self.buffer)
    
    def clear(self):
        """Clear all samples from buffer."""
        self.buffer.clear()
        logger.info("Replay buffer cleared")
    
    def save_to_disk(self):
        """Persist buffer to disk."""
        try:
            samples_data = []
            for i, sample in enumerate(self.buffer):
                # Save image
                img_path = self.buffer_path / f"sample_{i}_{sample.frame_id}.jpg"
                img_path.write_bytes(sample.image_data)
                
                # Save metadata
                samples_data.append({
                    "image_path": str(img_path),
                    "original_prediction": sample.original_prediction,
                    "corrected_label": sample.corrected_label,
                    "confidence": sample.confidence,
                    "timestamp": sample.timestamp,
                    "frame_id": sample.frame_id
                })
            
            # Save metadata JSON
            metadata_path = self.buffer_path / "buffer_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(samples_data, f, indent=2)
            
            logger.info(f"Buffer saved to disk: {len(samples_data)} samples")
            
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}")
    
    def load_from_disk(self):
        """Load buffer from disk."""
        try:
            metadata_path = self.buffer_path / "buffer_metadata.json"
            if not metadata_path.exists():
                return
            
            with open(metadata_path, 'r') as f:
                samples_data = json.load(f)
            
            for data in samples_data:
                img_path = Path(data["image_path"])
                if img_path.exists():
                    sample = TrainingSample(
                        image_data=img_path.read_bytes(),
                        original_prediction=data["original_prediction"],
                        corrected_label=data["corrected_label"],
                        confidence=data["confidence"],
                        timestamp=data["timestamp"],
                        frame_id=data["frame_id"]
                    )
                    self.buffer.append(sample)
            
            logger.info(f"Loaded {len(self.buffer)} samples from disk")
            
        except Exception as e:
            logger.error(f"Failed to load buffer: {e}")
    
    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        label_counts = {}
        for sample in self.buffer:
            label_counts[sample.corrected_label] = label_counts.get(sample.corrected_label, 0) + 1
        
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "utilization": len(self.buffer) / self.max_size if self.max_size > 0 else 0,
            "label_distribution": label_counts
        }


class PlanktonDataset(Dataset):
    """PyTorch dataset for training samples."""
    
    def __init__(self, samples: List[TrainingSample], species_classes: List[str]):
        self.samples = samples
        self.species_classes = species_classes
        self.label_to_idx = {label: idx for idx, label in enumerate(species_classes)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Decode image
        nparr = np.frombuffer(sample.image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess (match model preprocessing)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        
        # Get label index
        label_idx = self.label_to_idx.get(sample.corrected_label, 0)
        
        return image_tensor, label_idx


class TrainingManager:
    """Manages model training and fine-tuning."""
    
    def __init__(self):
        self.replay_buffer = ReplayBuffer()
        self.training_active = False
        self.training_history = []
        self.current_job: Optional[Dict] = None
    
    async def submit_correction(
        self,
        image_data: bytes,
        original_prediction: str,
        corrected_label: str,
        confidence: float,
        frame_id: str
    ) -> Dict:
        """
        Submit a corrected label for training.
        
        Args:
            image_data: JPEG image bytes
            original_prediction: Original model prediction
            corrected_label: User-corrected label
            confidence: Original prediction confidence
            frame_id: Unique frame identifier
            
        Returns:
            Status dictionary
        """
        try:
            sample = TrainingSample(
                image_data=image_data,
                original_prediction=original_prediction,
                corrected_label=corrected_label,
                confidence=confidence,
                timestamp=time.time(),
                frame_id=frame_id
            )
            
            self.replay_buffer.add(sample)
            self.replay_buffer.save_to_disk()
            
            buffer_stats = self.replay_buffer.get_stats()
            
            return {
                "status": "success",
                "message": "Correction submitted",
                "buffer_size": buffer_stats["size"],
                "buffer_utilization": buffer_stats["utilization"]
            }
            
        except Exception as e:
            logger.error(f"Failed to submit correction: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def start_training(self) -> Dict:
        """
        Start fine-tuning the model on buffered samples.
        Runs in background task.
        
        Returns:
            Job status dictionary
        """
        if self.training_active:
            return {
                "status": "error",
                "message": "Training already in progress"
            }
        
        samples = self.replay_buffer.get_all()
        if len(samples) < 10:
            return {
                "status": "error",
                "message": f"Insufficient samples for training. Need at least 10, have {len(samples)}"
            }
        
        # Create training job
        job_id = f"train_{int(time.time())}"
        self.current_job = {
            "job_id": job_id,
            "status": "starting",
            "started_at": time.time(),
            "samples_count": len(samples),
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": TRAINING_CONFIG["epochs"]
        }
        
        # Start training in background
        asyncio.create_task(self._train_model(samples, job_id))
        
        return {
            "status": "success",
            "message": "Training started",
            "job_id": job_id
        }
    
    async def _train_model(self, samples: List[TrainingSample], job_id: str):
        """
        Background task for model training.
        Only fine-tunes the classifier head (last layer).
        """
        self.training_active = True
        model = get_model()
        
        try:
            logger.info(f"Starting training job {job_id} with {len(samples)} samples")
            
            # Update job status
            self.current_job["status"] = "training"
            
            # Create dataset and dataloader
            dataset = PlanktonDataset(samples, SPECIES_CLASSES)
            
            # Split into train/val
            val_size = int(len(dataset) * TRAINING_CONFIG["validation_split"])
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAINING_CONFIG["batch_size"],
                shuffle=True,
                num_workers=0  # Use 0 for Windows compatibility
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=TRAINING_CONFIG["batch_size"],
                shuffle=False,
                num_workers=0
            )
            
            # Freeze backbone if configured
            if TRAINING_CONFIG["freeze_backbone"]:
                for param in model.model.parameters():
                    param.requires_grad = False
                # Unfreeze only the final classifier layer
                for param in model.model.fc.parameters():
                    param.requires_grad = True
            
            # Setup optimizer and loss
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.model.parameters()),
                lr=TRAINING_CONFIG["learning_rate"]
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(TRAINING_CONFIG["epochs"]):
                self.current_job["current_epoch"] = epoch + 1
                
                # Train
                model.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for images, labels in train_loader:
                    images = images.to(model.device)
                    labels = labels.to(model.device)
                    
                    optimizer.zero_grad()
                    outputs = model.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                train_acc = train_correct / train_total if train_total > 0 else 0
                
                # Validate
                model.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(model.device)
                        labels = labels.to(model.device)
                        
                        outputs = model.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_acc = val_correct / val_total if val_total > 0 else 0
                
                # Update progress
                progress = (epoch + 1) / TRAINING_CONFIG["epochs"]
                self.current_job["progress"] = progress
                self.current_job["train_acc"] = train_acc
                self.current_job["val_acc"] = val_acc
                
                logger.info(
                    f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}: "
                    f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}"
                )
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    model.save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= TRAINING_CONFIG.get("early_stopping_patience", 3):
                        logger.info("Early stopping triggered")
                        break
            
            # Check if new model meets minimum accuracy
            if best_val_acc < TRAINING_CONFIG["min_accuracy_threshold"]:
                logger.warning(
                    f"New model accuracy {best_val_acc:.3f} below threshold "
                    f"{TRAINING_CONFIG['min_accuracy_threshold']:.3f}. Rolling back."
                )
                model.rollback()
                self.current_job["status"] = "failed"
                self.current_job["message"] = "Model accuracy below threshold, rolled back"
            else:
                # Reload model to apply changes
                model.load_model()
                self.current_job["status"] = "completed"
                self.current_job["message"] = f"Training completed. Best val acc: {best_val_acc:.3f}"
                self.current_job["best_val_acc"] = best_val_acc
            
            self.current_job["completed_at"] = time.time()
            self.training_history.append(self.current_job.copy())
            
            logger.info(f"Training job {job_id} completed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.current_job["status"] = "failed"
            self.current_job["message"] = str(e)
            self.current_job["completed_at"] = time.time()
            
        finally:
            self.training_active = False
            # Restore model to eval mode
            model.model.eval()
    
    def get_training_status(self) -> Dict:
        """Get current training job status."""
        if self.current_job is None:
            return {
                "status": "idle",
                "message": "No training job active"
            }
        return self.current_job.copy()
    
    def get_buffer_stats(self) -> Dict:
        """Get replay buffer statistics."""
        return self.replay_buffer.get_stats()
    
    def clear_buffer(self):
        """Clear the replay buffer."""
        self.replay_buffer.clear()


# Global training manager instance
training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    """Get global training manager instance."""
    return training_manager
