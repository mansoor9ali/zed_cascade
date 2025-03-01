import pyzed.sl as sl
import numpy as np
from typing import Tuple, Optional, Dict

def initialize_camera(svo: str = None) -> Tuple[sl.Camera, sl.CameraInformation]:
    """
    Initialize the ZED camera.
    
    Args:
        svo (str, optional): Path to the SVO file. Defaults to None.
    
    Returns:
        Tuple[sl.Camera, sl.CameraInformation]: Initialized ZED camera and its information.
    """
    zed = sl.Camera()
    input_type = sl.InputType()
    if svo:
        input_type.set_from_svo_file(svo)
    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50
    if not svo:
        init_params.camera_resolution = sl.RESOLUTION.HD1080
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Camera initialization failed: {status}")
    camera_info = zed.get_camera_information()
    return zed, camera_info

def retrieve_frame(zed: sl.Camera, point_cloud: sl.Mat, camera_res: sl.Resolution) -> Tuple[np.ndarray, sl.Mat]:
    """
    Retrieve frame and point cloud data from the ZED camera.
    
    Args:
        zed (sl.Camera): ZED camera instance.
        point_cloud (sl.Mat): Point cloud matrix.
        camera_res (sl.Resolution): Camera resolution.
    
    Returns:
        Tuple[np.ndarray, sl.Mat]: Retrieved image and point cloud data.
    """
    image_left = sl.Mat()
    zed.retrieve_image(image_left, sl.VIEW.LEFT)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, camera_res)
    image_net = image_left.get_data()
    return image_net, point_cloud