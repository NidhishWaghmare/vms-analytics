# VMS Analytics - Video Management System

## Overview

VMS Analytics is a comprehensive real-time video analytics system built for vehicle detection, tracking, and Automatic Number Plate Recognition (ANPR). The system processes RTSP camera streams to monitor vehicle entry/exit and provides intelligent analytics through REST API integration.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Requirements](#system-requirements)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Performance Metrics](#performance-metrics)
- [Process Flow](#process-flow)
- [Memory & CPU Usage](#memory--cpu-usage)
- [Troubleshooting](#troubleshooting)
- [Service Management](#service-management)

## Architecture Overview

The VMS Analytics system operates as a multi-threaded application that:
1. Fetches camera configurations from a REST API
2. Processes multiple RTSP camera streams concurrently
3. Performs real-time object detection using YOLO models
4. Tracks vehicles using DeepSORT algorithm
5. Detects number plates using specialized ANPR models
6. Sends analytics results back to the management system

## Key Features

### Core Functionality
- **Real-time Vehicle Detection**: Uses YOLOv8 for detecting cars, buses, and trucks
- **Vehicle Tracking**: DeepSORT-based multi-object tracking across frames
- **ANPR (Automatic Number Plate Recognition)**: Specialized model for license plate detection
- **Line Crossing Detection**: Monitors vehicle entry/exit across defined ROI lines
- **Multi-Camera Support**: Concurrent processing of multiple camera streams
- **Error Reporting**: Automatic error reporting with confirmation-based retry mechanism

### Advanced Features
- **ROI Configuration**: Region of Interest configuration per camera
- **Direction Swapping**: Special handling for specific cameras
- **Frame Skipping**: Optimized processing with configurable frame skip rates
- **Screenshot Capture**: Automatic screenshot capture for events
- **Time-based Confidence**: Dynamic confidence thresholds based on time of day
- **Event Deduplication**: Prevents duplicate events within configurable time windows

## Technology Stack

### Core Technologies
- **Python 3.12**: Primary programming language
- **OpenCV 4.10.0+**: Computer vision and image processing
- **PyTorch**: Deep learning framework (via Ultralytics)
- **Ultralytics YOLOv8**: Object detection models
- **DeepSORT**: Multi-object tracking algorithm

### Libraries & Dependencies
```
opencv-python>=4.10.0       # Computer vision operations
numpy>=1.26.0               # Numerical computing
ultralytics>=8.3.0          # YOLO model implementation
requests>=2.32.0            # HTTP API communication
deep-sort-realtime          # Object tracking
psutil>=6.0.0              # System monitoring
pathlib>=1.0.1             # File path operations
pyyaml>=6.0.1              # Configuration files
```

### Models Used
- **YOLOv8x**: Primary vehicle detection model (yolov8x.pt)
- **ANPR Model**: Custom number plate recognition model (ANPR.pt)
- **Backup Models**: yolov8s.pt, best.pt for various configurations

## System Requirements

### Operating System
- **Linux (Ubuntu 20.04+ recommended)**
- **Python 3.8-3.12**

### Minimum Hardware Requirements
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: NVIDIA GPU with 4GB VRAM (optional but recommended)
- **Storage**: 50 GB free space for models, logs, and screenshots
- **Network**: Stable internet connection for RTSP streams

### Recommended Hardware Requirements
- **CPU**: 8+ cores, 3.0+ GHz (Intel i7/AMD Ryzen 7)
- **RAM**: 32 GB for multiple camera processing
- **GPU**: NVIDIA RTX 3060 or better with 8GB+ VRAM
- **Storage**: 500 GB SSD for optimal I/O performance
- **Network**: Dedicated network interface for camera streams

## Hardware Requirements

### CPU Requirements
- **Minimum**: Intel Core i5 or AMD Ryzen 5
- **Recommended**: Intel Core i7-10700K or AMD Ryzen 7 3700X
- **Cores**: 4-8 cores for optimal multi-camera processing
- **Architecture**: x86_64 architecture required

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU with CUDA support**
- **Minimum VRAM**: 4 GB
- **Recommended VRAM**: 8+ GB for multiple camera streams
- **Supported Architectures**: Pascal, Turing, Ampere, Ada Lovelace
- **CUDA Compute Capability**: 6.0 or higher

### Memory Requirements
- **System RAM**: 8-32 GB depending on camera count
- **Per Camera**: ~1-2 GB RAM per active camera stream
- **Model Loading**: ~2-4 GB for YOLO models
- **Buffer Memory**: Additional memory for frame buffering and processing

### Storage Requirements
- **Model Files**: ~500 MB for YOLO models
- **Log Files**: 10 MB per log file (5 backups)
- **Screenshots**: Variable based on event frequency
- **Operating System**: 20+ GB free space recommended

## Installation

### 1. Clone Repository
```bash
cd /home/ubantu
git clone <repository-url> vms
cd vms
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
Place the following model files in the `models/` directory:
- `yolov8x.pt` - Main vehicle detection model
- `ANPR.pt` - Number plate recognition model
- `yolov8s.pt` - Lightweight vehicle detection model
- `best.pt` - Custom trained model (if available)

### 5. Create Directory Structure
```bash
mkdir -p data/{logs,screenshots,roi_configs}
```

## Configuration

### Environment Variables
```bash
export API_IP="192.168.1.38:8001"  # Override default API endpoint
```

### Key Configuration Parameters
```python
# Performance Settings
MAX_THREADS = 4                    # Concurrent camera processing threads
FRAME_SKIP = 5                     # Process every 5th frame
FETCH_INTERVAL = 60                # Fetch new cameras every 60 seconds

# Detection Settings
CROSSING_TOLERANCE = 10            # Pixel tolerance for line crossing
SCREENSHOT_TIME_WINDOW = 5         # Seconds between screenshots
GLOBAL_EVENT_TIME_WINDOW = 1       # Event deduplication window

# Logging Settings
LOG_MAX_SIZE = 10 * 1024 * 1024   # 10 MB per log file
LOG_BACKUP_COUNT = 5               # Keep 5 log backups
```

### Model Configuration
```python
MODEL_CONFIG = {
    "1": {
        "name": "vehicleInOut",
        "classes": [2, 5, 7]  # car, bus, truck
    },
    "2": {
        "name": "anpr",
        "classes": [2, 5, 7]  # same classes for ANPR
    }
}
```

## API Endpoints

### Input API (Camera Configuration)
```
GET http://{API_IP}/api/v1/aiAnalytics/getCamerasForAnalytics
```

### Output API (Analytics Results)
```
POST http://{API_IP}/api/v1/aiAnalytics/sendAnalyticsJson
```

### Error Reporting API
```
POST http://{API_IP}/api/v1/aiAnalytics/reportError
```

## Performance Metrics

### CPU Usage
- **Base Load**: 10-20% per camera stream
- **Peak Load**: 60-80% during intensive processing
- **Optimization**: Frame skipping reduces CPU load by ~50%

### GPU Usage (if available)
- **VRAM Usage**: 2-4 GB for model loading + inference
- **GPU Utilization**: 30-70% depending on model complexity
- **Inference Speed**: 5-15ms per frame on modern GPUs

### Memory Usage
- **Base Memory**: ~2 GB for application startup
- **Per Camera**: 1-2 GB RAM per active stream
- **Peak Memory**: Can reach 8-16 GB with 4+ cameras
- **Memory Management**: Aggressive garbage collection implemented

### Network Usage
- **RTSP Streams**: 2-10 Mbps per camera (depending on resolution/quality)
- **API Communication**: <1 Mbps for analytics data
- **Total Bandwidth**: Scales linearly with camera count

## Process Flow

### 1. System Initialization
```
├── Load Configuration
├── Initialize Logging System
├── Load YOLO Models (Vehicle + ANPR)
├── Setup API Endpoints
└── Create Directory Structure
```

### 2. Camera Management Loop
```
├── Fetch Camera Configurations from API
├── Group Cameras by ID and Model Types
├── Start/Stop Camera Processing Threads
├── Handle Error Reporting and Confirmations
└── Wait for Next Fetch Interval
```

### 3. Camera Processing Pipeline (Per Camera)
```
├── Initialize Camera Stream (RTSP/File)
├── Load/Configure ROI Settings
├── Initialize DeepSORT Tracker
├── Frame Processing Loop:
│   ├── Read Frame from Stream
│   ├── Skip Frames (Optimization)
│   ├── Resize Frame if Needed
│   ├── Run YOLO Detection (Bottom 60% of frame)
│   ├── Update DeepSORT Tracker
│   ├── Detect Line Crossings
│   ├── Capture Screenshots for Events
│   ├── Send Analytics to API
│   └── Memory Cleanup
└── Cleanup on Exit
```

### 4. Event Processing
```
├── Vehicle Detection
├── Track Assignment/Update
├── Line Crossing Analysis
├── Event Type Classification (Enter/Exit)
├── ANPR Processing (if configured)
├── Screenshot Capture
├── JSON Log Creation
└── API Transmission
```

## Memory & CPU Usage

### Memory Breakdown
```
Component                   Memory Usage
─────────────────────────────────────────
YOLOv8x Model Loading       ~2.0 GB
ANPR Model Loading          ~0.5 GB
DeepSORT Tracker            ~0.2 GB per camera
Frame Buffers               ~0.1 GB per camera
Python Runtime              ~0.5 GB
OpenCV Operations           ~0.3 GB per camera
Application Logic           ~0.3 GB
─────────────────────────────────────────
Total (4 cameras)          ~8-12 GB
```

### CPU Usage Patterns
```
Operation                   CPU %    Notes
─────────────────────────────────────────────
YOLO Inference              40-60%   Per camera, per frame
DeepSORT Tracking           5-10%    Per camera
Frame Processing            10-15%   Resize, color conversion
Line Crossing Detection     <5%      Geometric calculations
API Communication           <2%      Network I/O
Memory Management           5-10%    Garbage collection
─────────────────────────────────────────────
Total (4 cameras)          70-90%   On 8-core system
```

### GPU Utilization (NVIDIA)
```
Operation                   GPU %    VRAM Usage
───────────────────────────────────────────────
Model Loading               -        3-4 GB
YOLO Inference              30-70%   +0.5 GB per stream
Memory Transfers            10-20%   Variable
CUDA Operations             5-15%    Background processes
───────────────────────────────────────────────
Total (4 cameras)          50-85%   4-6 GB VRAM
```

### Performance Optimization Features
- **Frame Skipping**: Reduces CPU load by processing every Nth frame
- **Memory Management**: Aggressive garbage collection after each frame
- **ROI Processing**: Detection limited to bottom 60% of frame
- **Dynamic Thresholds**: Time-based confidence adjustment
- **Concurrent Processing**: Thread pool for multiple cameras
- **Smart Event Deduplication**: Prevents redundant processing

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Monitor memory usage
htop
# Check Python memory
ps aux | grep python
# Review log files for memory leaks
tail -f data/logs/analytics.log
```

#### Camera Connection Issues
```bash
# Test RTSP stream manually
ffplay rtsp://camera-ip/stream
# Check network connectivity
ping camera-ip
# Review error logs
grep "Failed to open stream" data/logs/analytics.log
```

#### Model Loading Errors
```bash
# Verify model files exist
ls -la models/
# Check file permissions
chmod 644 models/*.pt
# Validate model integrity
python -c "from ultralytics import YOLO; YOLO('models/yolov8x.pt')"
```

#### API Communication Issues
```bash
# Test API endpoints
curl -X GET http://{API_IP}/api/v1/aiAnalytics/getCamerasForAnalytics
# Check network connectivity
ping {API_IP}
# Review API logs
grep "Failed to" data/logs/analytics.log
```

### Performance Tuning

#### For CPU-Limited Systems
```python
# Reduce concurrent threads
MAX_THREADS = 2

# Increase frame skipping
FRAME_SKIP = 10

# Use lighter model
# Replace yolov8x.pt with yolov8s.pt
```

#### For Memory-Limited Systems
```python
# Reduce screenshot time window
SCREENSHOT_TIME_WINDOW = 10

# Increase garbage collection frequency
# Add more gc.collect() calls
```

#### For Network-Limited Systems
```python
# Increase fetch interval
FETCH_INTERVAL = 120

# Reduce screenshot quality
# Modify cv2.imwrite parameters
```

## Service Management

### SystemD Service Configuration
The system includes a systemd service file (`vms-analytics.service`) for automatic startup and management.

#### Service Commands
```bash
# Start service
sudo systemctl start vms-analytics

# Stop service  
sudo systemctl stop vms-analytics

# Enable auto-start
sudo systemctl enable vms-analytics

# Check status
sudo systemctl status vms-analytics

# View logs
journalctl -u vms-analytics -f
```

#### Service Configuration
```ini
[Unit]
Description=VMS Analytics Service
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=ubantu
WorkingDirectory=/home/ubantu/vms
Environment=PATH=/home/ubantu/vms/venv/bin
ExecStart=/home/ubantu/vms/venv/bin/python /home/ubantu/vms/main.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vms-analytics

[Install]
WantedBy=multi-user.target
```

### Manual Execution
```bash
# Activate virtual environment
source venv/bin/activate

# Run application
python main.py

# Run with custom API IP
API_IP="custom-ip:port" python main.py
```

## Monitoring and Maintenance

### Log Files
- **Location**: `data/logs/analytics.log`
- **Rotation**: 10 MB per file, 5 backups
- **Format**: Timestamp - Level - Message

### Screenshots
- **Location**: `data/screenshots/`
- **Naming**: `{event_type}_{camera_id}_{timestamp}.jpg`
- **Types**: enter_, exit_, plate_enter_

### ROI Configurations
- **Location**: `data/roi_configs/`
- **Format**: JSON files per camera
- **Structure**: `{camera_id}_roi.json`

### Health Monitoring
```bash
# Check process status
ps aux | grep main.py

# Monitor resource usage
htop

# Check disk space
df -h

# Monitor network usage
iftop
```

## License and Support

This VMS Analytics system is designed for production use in video surveillance environments. For technical support, please review the logs first and ensure all system requirements are met.

### File Structure
```
vms/
├── main.py                    # Main application
├── requirements.txt           # Python dependencies
├── vms-analytics.service      # SystemD service file
├── README.md                  # This documentation
├── data/
│   ├── logs/                  # Application logs
│   ├── screenshots/           # Event screenshots
│   └── roi_configs/           # Camera ROI configurations
├── models/                    # YOLO model files
│   ├── yolov8x.pt
│   ├── ANPR.pt
│   └── ...
└── venv/                      # Python virtual environment
```

---

*Last Updated: August 2025*
*Version: 1.0*
