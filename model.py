import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2
import pyzed.sl as sl

def initialize_model(weights: str) -> YOLO:
    """
    Initialize the YOLO model with given weights.
    
    Args:
        weights (str): Path to the YOLO model weights.
    
    Returns:
        YOLO: Initialized YOLO model.
    """
    model = YOLO(weights)
    return model

def process_detections(model: YOLO, det, image: np.ndarray, point_cloud: sl.Mat) -> np.ndarray:
    """
    Process and annotate detections on the image.
    
    Args:
        model (YOLO): YOLO model.
        det: Detection results from the model.
        image (np.ndarray): Original image.
        point_cloud (sl.Mat): Point cloud data.
    
    Returns:
        np.ndarray: Annotated image.
    """
    annotator = Annotator(image)
    for r in det:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            class_id = int(box.cls)
            color = colors(class_id, True)
            conf = box.conf.item()
            class_label = model.names[class_id]
            label = f"{class_label} {conf:.2f}"
            annotator.box_label([x1, y1, x2, y2], label=label, color=color)
            cv2.circle(annotator.im, center, 4, color, -1)
            _, pc_value = point_cloud.get_value(center[0], center[1])
            x, y, z = pc_value[0], pc_value[1], pc_value[2]
            xyz_label = f"X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
            cv2.putText(annotator.im, xyz_label, (x1 - 150, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return annotator.im