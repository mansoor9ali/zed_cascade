"""
ZED Camera Module
----------------
This module handles all ZED camera operations including initialization,
frame retrieval, and camera configuration. It provides a high-level interface
for working with the ZED stereo camera.

Key Features:
- Camera initialization with custom parameters
- Frame retrieval with various view options
- Point cloud data extraction
- Camera calibration information
- Runtime parameter management
"""

import pyzed.sl as sl
import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Configuration class for ZED camera parameters.
    
    Attributes:
        resolution (sl.RESOLUTION): Camera resolution (default: HD1080)
        depth_mode (sl.DEPTH_MODE): Depth mode (default: ULTRA)
        coordinate_system (sl.COORDINATE_SYSTEM): Coordinate system (default: RIGHT_HANDED_Y_UP)
        depth_max_distance (float): Maximum depth distance in meters (default: 50)
        depth_min_distance (float): Minimum depth distance in meters (default: 0.3)
        enable_image_enhancement (bool): Enable image enhancement (default: True)
    """
    resolution: sl.RESOLUTION = sl.RESOLUTION.HD1080
    depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.ULTRA
    coordinate_system: sl.COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    depth_max_distance: float = 50.0
    depth_min_distance: float = 0.3
    enable_image_enhancement: bool = True

def initialize_camera(svo: Optional[str] = None, config: Optional[CameraConfig] = None) -> Tuple[sl.Camera, sl.CameraInformation]:
    """Initialize the ZED camera with specified configuration.
    
    Args:
        svo (str, optional): Path to the SVO file for playback. Defaults to None.
        config (CameraConfig, optional): Camera configuration. Defaults to None.
    
    Returns:
        Tuple[sl.Camera, sl.CameraInformation]: Initialized ZED camera and its information.
    
    Raises:
        RuntimeError: If camera initialization fails.
        ValueError: If invalid configuration parameters are provided.
    """
    zed = sl.Camera()
    if config is None:
        config = CameraConfig()

    try:
        input_type = sl.InputType()
        if svo:
            input_type.set_from_svo_file(svo)

        init_params = sl.InitParameters(input_t=input_type)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = config.depth_mode
        init_params.coordinate_system = config.coordinate_system
        init_params.depth_maximum_distance = config.depth_max_distance
        init_params.depth_minimum_distance = config.depth_min_distance
        init_params.enable_image_enhancement = config.enable_image_enhancement

        if not svo:
            init_params.camera_resolution = config.resolution

        logger.info(f"Initializing camera with resolution: {config.resolution}")
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Camera initialization failed: {status}")

        camera_info = zed.get_camera_information()
        logger.info(f"Camera initialized successfully. Serial number: {camera_info.serial_number}")
        return zed, camera_info

    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        raise

def retrieve_frame(zed: sl.Camera, point_cloud: sl.Mat, camera_res: sl.Resolution,
                  view_mode: sl.VIEW = sl.VIEW.LEFT) -> Tuple[np.ndarray, sl.Mat]:
    """Retrieve frame and point cloud data from the ZED camera.
    
    Args:
        zed (sl.Camera): ZED camera instance.
        point_cloud (sl.Mat): Point cloud matrix.
        camera_res (sl.Resolution): Camera resolution.
        view_mode (sl.VIEW, optional): Camera view mode. Defaults to LEFT.
    
    Returns:
        Tuple[np.ndarray, sl.Mat]: Retrieved image and point cloud data.
    
    Raises:
        RuntimeError: If frame retrieval fails.
    """
    try:
        image = sl.Mat()
        zed.retrieve_image(image, view_mode)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, camera_res)
        return image.get_data(), point_cloud
    except Exception as e:
        logger.error(f"Error retrieving frame: {e}")
        raise RuntimeError(f"Frame retrieval failed: {e}")

def get_depth_at_point(point_cloud: sl.Mat, x: int, y: int) -> Tuple[float, float, float]:
    """Get depth information at a specific point in the point cloud.
    
    Args:
        point_cloud (sl.Mat): Point cloud matrix.
        x (int): X coordinate in the image.
        y (int): Y coordinate in the image.
    
    Returns:
        Tuple[float, float, float]: X, Y, Z coordinates in meters.
    """
    err, point3D = point_cloud.get_value(x, y)
    if err == sl.ERROR_CODE.SUCCESS:
        return point3D[0], point3D[1], point3D[2]
    return 0.0, 0.0, 0.0

def get_camera_params(camera_info: sl.CameraInformation) -> Dict[str, Union[float, str, int]]:
    """Get camera parameters and specifications.
    
    Args:
        camera_info (sl.CameraInformation): Camera information object.
    
    Returns:
        Dict[str, Union[float, str, int]]: Dictionary containing camera parameters.
    """
    params = {
        'serial_number': camera_info.serial_number,
        'firmware_version': camera_info.camera_configuration.firmware_version,
        'resolution': {
            'width': camera_info.camera_resolution.width,
            'height': camera_info.camera_resolution.height
        },
        'fps': camera_info.camera_fps,
        'focal_length': {
            'left': camera_info.calibration_parameters.left_cam.focal_length,
            'right': camera_info.calibration_parameters.right_cam.focal_length
        }
    }
    return params