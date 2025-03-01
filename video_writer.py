import cv2
from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Dict

def create_video_writer(save_path: str, fourcc: str, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create a video writer for saving the output video.
    
    Args:
        save_path (str): Path to save the output video.
        fourcc (str): FourCC code for the codec.
        fps (int): Frames per second.
        frame_size (Tuple[int, int]): Frame size (width, height).
    
    Returns:
        cv2.VideoWriter: Initialized video writer.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    return video_writer

def write_frame(video_writer: cv2.VideoWriter, frame: np.ndarray) -> None:
    """
    Write a frame to the video writer.
    
    Args:
        video_writer (cv2.VideoWriter): Video writer instance.
        frame (np.ndarray): Frame to write.
    """
    video_writer.write(frame)