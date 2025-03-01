"""
YOLO Model Module
----------------
This module handles all YOLO model operations including initialization,
inference, and visualization. It provides a comprehensive interface for
object detection using the YOLO model with ZED camera integration.

Key Features:
- Model initialization and management
- Detection processing and visualization
- 3D position estimation
- Detection filtering and tracking
- Performance metrics
"""

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
import pyzed.sl as sl
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration for detection parameters.
    
    Attributes:
        conf_threshold (float): Confidence threshold for detections
        iou_threshold (float): IOU threshold for NMS
        max_det (int): Maximum number of detections per frame
        classes (Optional[List[int]]): List of classes to detect, None for all
        track (bool): Enable object tracking
    """
    conf_threshold: float = 0.4
    iou_threshold: float = 0.45
    max_det: int = 100
    classes: Optional[List[int]] = None
    track: bool = False

class ModelMetrics:
    """Class to track model performance metrics."""
    def __init__(self):
        self.inference_times = []
        self.fps_history = []
        self.last_time = time.time()
    
    def update(self, inference_time: float):
        """Update metrics with new inference time."""
        self.inference_times.append(inference_time)
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        # Keep only recent history
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
            self.fps_history.pop(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'avg_fps': np.mean(self.fps_history),
            'min_fps': np.min(self.fps_history),
            'max_fps': np.max(self.fps_history)
        }

def initialize_model(weights: str, device: Optional[str] = None) -> YOLO:
    """Initialize the YOLO model with given weights.
    
    Args:
        weights (str): Path to the YOLO model weights.
        device (str, optional): Device to run model on ('cpu', 'cuda', etc.).
    
    Returns:
        YOLO: Initialized YOLO model.
        
    Raises:
        RuntimeError: If model initialization fails.
    """
    try:
        logger.info(f"Initializing YOLO model from {weights}")
        if device:
            torch.device(device)
        model = YOLO(weights)
        logger.info(f"Model initialized successfully. Classes: {len(model.names)}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

def process_detections(
    model: YOLO,
    det,
    image: np.ndarray,
    point_cloud: sl.Mat,
    config: Optional[DetectionConfig] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """Process and annotate detections on the image.
    
    Args:
        model (YOLO): YOLO model.
        det: Detection results from the model.
        image (np.ndarray): Original image.
        point_cloud (sl.Mat): Point cloud data.
        config (DetectionConfig, optional): Detection configuration.
    
    Returns:
        Tuple[np.ndarray, List[Dict]]: Annotated image and list of detection info.
    """
    if config is None:
        config = DetectionConfig()
        
    annotator = Annotator(image)
    detections = []
    
    try:
        for r in det:
            boxes = r.boxes
            for box in boxes:
                # Get detection coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Get detection info
                class_id = int(box.cls)
                conf = box.conf.item()
                
                # Filter by confidence and class
                if conf < config.conf_threshold:
                    continue
                if config.classes and class_id not in config.classes:
                    continue
                
                # Get color and label
                color = colors(class_id, True)
                class_label = model.names[class_id]
                label = f"{class_label} {conf:.2f}"
                
                # Draw bounding box and label
                annotator.box_label([x1, y1, x2, y2], label=label, color=color)
                cv2.circle(annotator.im, center, 4, color, -1)
                
                # Get 3D position
                _, pc_value = point_cloud.get_value(center[0], center[1])
                x, y, z = pc_value[0], pc_value[1], pc_value[2]
                
                # Add 3D position label
                xyz_label = f"X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                cv2.putText(annotator.im, xyz_label, 
                          (x1 - 150, y2 + 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Store detection info
                detections.append({
                    'class_id': class_id,
                    'class_name': class_label,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'center': center,
                    'position_3d': {'x': x, 'y': y, 'z': z}
                })
                
        return annotator.im, detections
        
    except Exception as e:
        logger.error(f"Error processing detections: {e}")
        raise RuntimeError(f"Detection processing failed: {e}")

def filter_detections(
    detections: List[Dict],
    min_conf: float = 0.5,
    max_depth: float = 10.0,
    classes: Optional[List[int]] = None
) -> List[Dict]:
    """Filter detections based on various criteria.
    
    Args:
        detections (List[Dict]): List of detection dictionaries.
        min_conf (float): Minimum confidence threshold.
        max_depth (float): Maximum depth in meters.
        classes (List[int], optional): List of class IDs to keep.
    
    Returns:
        List[Dict]: Filtered detections.
    """
    filtered = []
    for det in detections:
        if det['confidence'] < min_conf:
            continue
        if abs(det['position_3d']['z']) > max_depth:
            continue
        if classes and det['class_id'] not in classes:
            continue
        filtered.append(det)
    return filtered