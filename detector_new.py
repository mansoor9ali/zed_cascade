import argparse
import torch
import cv2
import numpy as np
import pyzed.sl as sl
from threading import Lock, Thread, Event
from queue import Queue, Full
from time import sleep
import logging
import yaml
from typing import Tuple, Optional, Dict
from model import initialize_model, process_detections
from camera import initialize_camera, retrieve_frame
from video_writer import create_video_writer, write_frame
from pathlib import Path
from ultralytics.utils.plotting import Annotator, colors
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionConfig:
    """Configuration class for detection parameters"""
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Default configuration
        self.weights = 'yolov8m.pt'
        self.svo = None
        self.img_size = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.45
        self.save_path = 'mydata/experiment/video/output.mp4'
        self.max_queue_size = 10
        self.fps = 20
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class ObjectDetector:
    """Main class for object detection using ZED camera and YOLO"""
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.lock = Lock()
        self.frame_queue = Queue(maxsize=config.max_queue_size)
        self.exit_event = Event()
        self.detection_ready = Event()
        self.annotator = None
        self.model = None
        self.zed = None
        self.video_writer = None

    def initialize_model(self) -> None:
        """Initialize YOLO model"""
        try:
            logger.info("Initializing YOLO model...")
            self.model = initialize_model(self.config.weights)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def initialize_camera(self) -> None:
        """Initialize ZED camera"""
        try:
            logger.info("Initializing ZED camera...")
            self.zed, camera_info = initialize_camera(self.config.svo)
            logger.info("Camera initialized successfully")
            return camera_info
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    def initialize_video_writer(self, camera_res) -> None:
        """Initialize video writer"""
        try:
            self.video_writer = create_video_writer(
                self.config.save_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self.config.fps,
                (camera_res.width, camera_res.height)
            )
            logger.info(f"Video writer initialized: {self.config.save_path}")
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            raise

    def detection_thread(self) -> None:
        """Thread for running object detection"""
        try:
            self.initialize_model()
            
            while not self.exit_event.is_set():
                if not self.frame_queue.empty():
                    with self.lock:
                        image_net, point_cloud = self.frame_queue.get()
                        
                    img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)
                    det = self.model.predict(
                        img, 
                        save=False, 
                        imgsz=self.config.img_size,
                        conf=self.config.conf_thres,
                        iou=self.config.iou_thres
                    )
                    
                    image = det[0].orig_img.copy()
                    annotated_image = process_detections(self.model, det, image, point_cloud)
                    
                    with self.lock:
                        self.annotator = Annotator(annotated_image)
                        self.detection_ready.set()
                
                sleep(0.001)  # Prevent CPU overuse
        except Exception as e:
            logger.error(f"Error in detection thread: {e}")
            self.exit_event.set()

    def capture_and_display(self) -> None:
        """Main loop for capturing and displaying frames"""
        try:
            camera_info = self.initialize_camera()
            camera_res = camera_info.camera_resolution
            self.initialize_video_writer(camera_res)
            
            image_left = sl.Mat()
            point_cloud = sl.Mat(camera_res.width, camera_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
            runtime_params = sl.RuntimeParameters()
            
            logger.info("Starting capture loop...")
            while not self.exit_event.is_set():
                if self.zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, camera_res)
                    
                    try:
                        self.frame_queue.put((image_left.get_data(), point_cloud), timeout=1)
                    except Full:
                        logger.warning("Frame queue is full, skipping frame")
                        continue
                    
                    if self.detection_ready.wait(timeout=1.0):
                        with self.lock:
                            if self.annotator is not None:
                                cv2.imshow("ZED Object Detection", self.annotator.im)
                                if self.video_writer is not None:
                                    write_frame(self.video_writer, self.annotator.im)
                        self.detection_ready.clear()
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.exit_event.set()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.zed is not None:
            self.zed.close()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Run the detection pipeline"""
        try:
            detection_thread = Thread(target=self.detection_thread)
            detection_thread.start()
            self.capture_and_display()
            detection_thread.join()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error running detection pipeline: {e}")
        finally:
            self.cleanup()

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ZED Object Detection with YOLO')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--weights', type=str, help='Path to YOLO weights')
    parser.add_argument('--svo', type=str, help='Path to SVO file')
    parser.add_argument('--img_size', type=int, help='Image size for inference')
    parser.add_argument('--conf_thres', type=float, help='Confidence threshold')
    parser.add_argument('--iou_thres', type=float, help='IOU threshold')
    parser.add_argument('--save', type=str, help='Path to save output video')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = DetectionConfig(
        config_path=args.config,
        **{k: v for k, v in vars(args).items() if v is not None}
    )
    with torch.no_grad():
        detector = ObjectDetector(config)
        detector.run()