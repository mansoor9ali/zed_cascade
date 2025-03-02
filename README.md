# ZED Camera Object Detection System

A sophisticated computer vision system using ZED stereo cameras and YOLOv8 for real-time object detection with 3D spatial information.

## 🔍 Project Overview

This project implements a real-time object detection system that combines:
- ZED stereo camera capabilities for depth sensing
- YOLOv8 for state-of-the-art object detection
- 3D spatial mapping of detected objects
- Multi-threaded processing for optimal performance

## 📁 Project Structure

```
zed_cascade/
├── detector.py     # Main detection pipeline
├── camera.py          # ZED camera operations
├── model.py           # YOLO model handling
├── video_writer.py    # Video output management
└── config.yaml        # Configuration settings
```

## 🛠 Key Features

### Camera Module (`camera.py`)
- Comprehensive ZED camera initialization and configuration
- Frame retrieval with various view options
- Point cloud data extraction
- Camera calibration and parameter management
- Depth measurement utilities

### Model Module (`model.py`)
- YOLO model initialization and management
- Real-time object detection processing
- 3D position estimation
- Performance metrics tracking
- Detection filtering capabilities

### Detection Pipeline (`detector_new.py`)
- Thread-safe implementation
- Efficient frame queue management
- Real-time video processing
- Configurable detection parameters
- Robust error handling

### Video Writer (`video_writer.py`)
- Flexible video output configuration
- Automatic directory creation
- Frame-by-frame writing capability
- Multiple codec support

## 🔧 Technical Improvements

1. **Enhanced Architecture**
   - Modular design with clear separation of concerns
   - Type-safe configuration management
   - Comprehensive error handling
   - Thread-safe operations

2. **Performance Optimization**
   - Multi-threaded processing
   - Efficient queue management
   - Performance metrics tracking
   - Memory-efficient operations

3. **Developer Experience**
   - Comprehensive documentation
   - Type hints throughout
   - Clear API interfaces
   - Flexible configuration options

4. **Error Handling**
   - Detailed error messages
   - Proper resource cleanup
   - Graceful failure handling
   - Logging system integration

## 🖥 Requirements

- Python 3.8+
- CUDA-compatible GPU
- ZED SDK
- Dependencies:
  * ultralytics
  * opencv-python
  * torch
  * pyzed
  * numpy
  * PyYAML

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure settings in `config.yaml`:
```yaml
weights: yolov8m.pt
img_size: 640
conf_thres: 0.4
iou_thres: 0.45
save_path: output/video.mp4
```

3. Run the detection pipeline:
```bash
python detector.py --config config.yaml
```

## ⚙️ Configuration Options

- `weights`: Path to YOLO model weights
- `svo`: Path to SVO file for playback (optional)
- `img_size`: Input image size for YOLO
- `conf_thres`: Confidence threshold
- `iou_thres`: IOU threshold for NMS
- `save_path`: Output video path
- `max_queue_size`: Frame queue size
- `fps`: Output video FPS

## 🎯 Future Improvements

- Advanced object tracking
- Multiple camera support
- Custom model training options
- Real-time data streaming
- GUI interface
- Performance optimizations

## ⚠️ Known Issues

- Ensure proper CUDA setup for optimal performance
- Video codec compatibility (using mp4v)
- Memory usage with high-resolution streams
- Thread synchronization in high-load scenarios

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
