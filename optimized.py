#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os
import multiprocessing as mp
import threading
import logging
import logging.handlers
import requests
import time
import gc
import concurrent.futures
import uuid
import json
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import signal
import sys

# Configuration
CROSSING_TOLERANCE = 10  # Pixels, tolerance for line crossing detection
MODEL_CONFIG = {
    "1": {"name": "vehicleInOut", "classes": [5, 7]}  # Classes: 5=bus, 7=truck
}
DEFAULT_API_IP = "192.168.1.38:8001"  # Fallback API IP
OUTPUT_API_ENDPOINT = "/api/v1/aiAnalytics/sendAnalyticsJson"
INPUT_API_ENDPOINT = "/api/v1/aiAnalytics/getCamerasForAnalytics"
ERROR_API_ENDPOINT = "/api/v1/aiAnalytics/reportError"
FETCH_INTERVAL = 60  # Fetch new camera data every hour (seconds)
MAX_PROCESSES = 6  # Increased to handle 3+ RTSP streams simultaneously
FRAME_SKIP = 3  # Process every 5th frame to reduce CPU usage
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 5  # Keep 5 log file backups
SCREENSHOT_TIME_WINDOW = 5  # Seconds to treat multiple detections as one
GLOBAL_EVENT_TIME_WINDOW = 1  # Seconds to deduplicate any event type
SPECIAL_CAMERA_DEDUP_WINDOW = 20  # Seconds for special camera deduplication
ENTRY_EXIT_DEDUP_WINDOW = 10  # Seconds to treat entry and exit as single detection
FRAME_BUFFER_SIZE = 10  # Store up to 10 frames to capture earliest frame
SPECIAL_CAMERA_IDS = [
    "uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d",  # Special camera with 20s deduplication - IGNORE ENTRY EVENTS
    "uuid:0f1d2d49-86cf-49d9-98fc-34750541b05d"   # Special camera with 20s deduplication - IGNORE EXIT EVENTS
]
# Camera that should treat entry and exit within 10 seconds as single detection
ENTRY_EXIT_DEDUP_CAMERA_ID = "uuid:afd04419-897a-4f6c-8108-7137d9a2c1b8"
# Camera-specific event filtering
IGNORE_ENTRY_CAMERA_ID = "uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d"  # Ignore entry events for this camera
IGNORE_EXIT_CAMERA_ID = "uuid:0f1d2d49-86cf-49d9-98fc-34750541b05d"   # Ignore exit events for this camera

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "screenshots"
JSON_DIR = DATA_DIR / "logs"
ROI_CONFIG_DIR = DATA_DIR / "roi_configs"
VEHICLE_MODEL_PATH = BASE_DIR / "models" / "yolov8x.pt"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)
ROI_CONFIG_DIR.mkdir(exist_ok=True)

# Setup logging with rotation
log_handler = logging.handlers.RotatingFileHandler(
    JSON_DIR / 'analytics.log',
    maxBytes=LOG_MAX_SIZE,
    backupCount=LOG_BACKUP_COUNT
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
if device == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    logger.warning("CUDA not available, falling back to CPU")

# Get API IP from environment variable or use default
API_IP = os.environ.get('API_IP', DEFAULT_API_IP)
INPUT_API_URL = f"http://{API_IP}{INPUT_API_ENDPOINT}"
OUTPUT_API_URL = f"http://{API_IP}{OUTPUT_API_ENDPOINT}"
ERROR_API_URL = f"http://{API_IP}{ERROR_API_ENDPOINT}"
logger.info(f"Input API URL: {INPUT_API_URL}")
logger.info(f"Output API URL: {OUTPUT_API_URL}")
logger.info(f"Error API URL: {ERROR_API_URL}")

def load_yolo_model():
    """Load YOLO model in each process to avoid sharing GPU memory across processes."""
    try:
        if not VEHICLE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Vehicle model file not found at {VEHICLE_MODEL_PATH}")
        vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
        # Move model to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            vehicle_model.to(device)
            print(f"Process {os.getpid()} - YOLO model loaded and moved to GPU: {device}")
        else:
            print(f"Process {os.getpid()} - YOLO model loaded on CPU: {device}")
        return vehicle_model, device
    except Exception as e:
        print(f"Process {os.getpid()} - Failed to load vehicle model: {e}")
        return None, None

# Global multiprocessing variables - will be initialized in main()
stop_event = None
fetch_trigger = None

def report_error(camera_id, error_message):
    """Send error details to the error API endpoint and wait for confirmation."""
    global fetch_trigger
    
    error_data = {
        "cameraId": camera_id,
        "errorMessage": str(error_message),
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M:%S"),
        "errorId": str(uuid.uuid4())
    }
    try:
        headers = {"Authorization": "Bearer YOUR_TOKEN_HERE"}
        response = requests.post(ERROR_API_URL, json=error_data, headers=headers, timeout=5)
        if response.status_code == 200:
            try:
                response_json = response.json()
                if response_json.get("status") == "received":
                    print(f"Process {os.getpid()} - Camera {camera_id} - Error report confirmed by frontend. Triggering immediate fetch.")
                    if fetch_trigger is not None:
                        fetch_trigger.set()
                else:
                    print(f"Process {os.getpid()} - Camera {camera_id} - Error report sent but no 'received' confirmation: {response_json}")
            except ValueError:
                print(f"Process {os.getpid()} - Camera {camera_id} - Invalid JSON response from error API: {response.text}")
        else:
            print(f"Process {os.getpid()} - Camera {camera_id} - Failed to report error: Status {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        print(f"Process {os.getpid()} - Camera {camera_id} - Failed to report error to {ERROR_API_URL}: {e}")
    finally:
        gc.collect()

def save_roi_config(camera_id, roi_points):
    """Save ROI points to a JSON file for the camera."""
    roi_file = ROI_CONFIG_DIR / f"{camera_id}_roi.json"
    roi_data = {"roi_points": roi_points}
    try:
        with open(roi_file, 'w') as f:
            json.dump(roi_data, f)
        print(f"Process {os.getpid()} - Camera {camera_id} - Saved ROI points to {roi_file}")
    except Exception as e:
        print(f"Process {os.getpid()} - Camera {camera_id} - Failed to save ROI points: {e}")

def load_roi_config(camera_id):
    """Load ROI points from a JSON file for the camera, if available."""
    roi_file = ROI_CONFIG_DIR / f"{camera_id}_roi.json"
    if roi_file.exists():
        try:
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
                roi_points = roi_data.get("roi_points", [])
                if len(roi_points) == 2:
                    print(f"Process {os.getpid()} - Camera {camera_id} - Loaded ROI points from {roi_file}")
                    return roi_points
        except Exception as e:
            print(f"Process {os.getpid()} - Camera {camera_id} - Failed to load ROI points: {e}")
    return None

def initialize_camera_state(model_id, camera_id):
    """Initialize state for a camera based on model ID."""
    base_state = {
        "roi_points": [],
        "roi_selected": False,
        "track_states": {},  # Track ID to state mapping
        "cap": None,
        "width": None,
        "height": None,
        "line_params": None,
        "frame_buffer": deque(maxlen=FRAME_BUFFER_SIZE)  # Initialize frame buffer
    }
    if model_id == "1":
        base_state.update({"enter_count": 0, "exit_count": 0})
    return base_state

def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def signed_distance(x, y, a, b, c):
    """Calculate signed distance from point (x, y) to line ax + by + c = 0."""
    denominator = np.sqrt(a**2 + b**2)
    if denominator == 0:
        return 0
    return (a * x + b * y + c) / denominator

def save_screenshot(frame, camera_id, timestamp, prefix):
    """Save a full frame screenshot and return its filename."""
    try:
        timestamp_str = timestamp.strftime("%d %b %Y %H:%M:%S").replace(" ", "_").replace(":", "_")
        filename = f"{prefix}{camera_id}_{timestamp_str}.jpg"
        filepath = OUTPUT_DIR / filename
        cv2.imwrite(str(filepath), frame)
        print(f"Process {os.getpid()} - Camera {camera_id} - Saved full frame screenshot: {filepath}")
        return filename
    except Exception as e:
        print(f"Process {os.getpid()} - Camera {camera_id} - Failed to save screenshot: {e}")
        return None
    finally:
        gc.collect()

def determine_crossing_vehicle(tracks, a, b, c, frame, camera_id, camera_info, state, shared_state):
    """Determine if vehicle centroids cross the ROI line and classify as Entering/Exiting."""
    try:
        # Get current timestamp
        timestamp = datetime.now()
        
        # Check if this is one of the special cameras that requires 20-second deduplication
        is_special_camera = camera_id in SPECIAL_CAMERA_IDS
        
        # Check if this is the camera that needs entry-exit deduplication
        is_entry_exit_dedup_camera = camera_id == ENTRY_EXIT_DEDUP_CAMERA_ID
        
        # For entry-exit deduplication camera, check if any event happened within 10 seconds
        if is_entry_exit_dedup_camera:
            last_any_event = shared_state["last_screenshot_timestamps"].get("last_any_event")
            if last_any_event is not None and (timestamp - last_any_event).total_seconds() <= ENTRY_EXIT_DEDUP_WINDOW:
                print(f"Process {os.getpid()} - Camera {camera_id} - Skipped vehicle event (within {ENTRY_EXIT_DEDUP_WINDOW}s of any previous event)")
                return None
        
        # For special cameras, check 20-second window; for others, use global event window
        elif is_special_camera:
            special_timestamp = shared_state["last_screenshot_timestamps"].get("special_camera_event")
            if special_timestamp is not None and (timestamp - special_timestamp).total_seconds() <= SPECIAL_CAMERA_DEDUP_WINDOW:
                print(f"Process {os.getpid()} - Camera {camera_id} - Skipped vehicle event (within {SPECIAL_CAMERA_DEDUP_WINDOW}s for special camera)")
                return None
        else:
            # Check global event timestamp to deduplicate across all event types
            global_timestamp = shared_state["last_screenshot_timestamps"].get("global_event")
            if global_timestamp is not None and (timestamp - global_timestamp).total_seconds() <= GLOBAL_EVENT_TIME_WINDOW:
                print(f"Process {os.getpid()} - Camera {camera_id} - Skipped vehicle event (within {GLOBAL_EVENT_TIME_WINDOW}s of any event)")
                return None

        log_entry = None
       
        print(f"Process {os.getpid()} - Camera {camera_id} - Processing {len(tracks)} tracks for crossing detection")
       
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            
            centroid = calculate_centroid(bbox)
            print(f"Process {os.getpid()} - Camera {camera_id} - Track {track_id} centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
           
            # Initialize track state if not present
            if track_id not in state['track_states']:
                state['track_states'][track_id] = {
                    'prev_centroid': None,
                    'has_crossed': False
                }
           
            track_state = state['track_states'][track_id]
            curr_centroid = centroid
            prev_centroid = track_state['prev_centroid']
           
            if prev_centroid is None:
                track_state['prev_centroid'] = curr_centroid
                continue
           
            prev_dist = signed_distance(prev_centroid[0], prev_centroid[1], a, b, c)
            curr_dist = signed_distance(curr_centroid[0], curr_centroid[1], a, b, c)
           
            event_type = None
            filename = None
           
            # Select earliest frame from buffer (first frame in deque, ~10 frames back when full)
            screenshot_frame = shared_state['frame_buffer'][0] if shared_state['frame_buffer'] else frame
           
            if not track_state['has_crossed']:
                # Left to right (entering)
                if prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
                    # Check if we should ignore entry events for this camera
                    if camera_id == IGNORE_ENTRY_CAMERA_ID:
                        print(f"Process {os.getpid()} - Camera {camera_id} - Vehicle {track_id} entered but entry event ignored (camera configured to ignore entries)")
                        track_state['prev_centroid'] = curr_centroid
                        continue
                   
                    state['enter_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "enter"
                    print(f"Process {os.getpid()} - Camera {camera_id} - Vehicle {track_id} entered. Total entering: {state['enter_count']}")
                    
                    # Handle screenshot and logging based on camera type
                    if is_entry_exit_dedup_camera:
                        # For entry-exit deduplication camera, always save screenshot and log since we already checked deduplication at function start
                        filename = save_screenshot(screenshot_frame, camera_id, timestamp, "enter_")
                        shared_state["last_screenshot_timestamps"]["last_any_event"] = timestamp
                        print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for enter event (entry-exit dedup camera)")
                    elif is_special_camera:
                        # For special camera, check 20-second window
                        last_timestamp = shared_state["last_screenshot_timestamps"].get("special_camera_event")
                        if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SPECIAL_CAMERA_DEDUP_WINDOW:
                            filename = save_screenshot(screenshot_frame, camera_id, timestamp, "enter_")
                            shared_state["last_screenshot_timestamps"]["special_camera_event"] = timestamp
                            print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for enter event (special camera)")
                        else:
                            print(f"Process {os.getpid()} - Camera {camera_id} - Skipped screenshot and log for enter event (within {SPECIAL_CAMERA_DEDUP_WINDOW}s for special camera)")
                            event_type = None  # Prevent log creation
                    else:
                        # For normal cameras, use original logic
                        last_timestamp = shared_state["last_screenshot_timestamps"].get("vehicle_enter")
                        if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SCREENSHOT_TIME_WINDOW:
                            filename = save_screenshot(screenshot_frame, camera_id, timestamp, "enter_")
                            shared_state["last_screenshot_timestamps"]["vehicle_enter"] = timestamp
                            shared_state["last_screenshot_timestamps"]["global_event"] = timestamp
                            print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for enter event")
                        else:
                            print(f"Process {os.getpid()} - Camera {camera_id} - Skipped screenshot and log for enter event (within {SCREENSHOT_TIME_WINDOW}s)")
                            event_type = None  # Prevent log creation
                            
                # Right to left (exiting)
                elif prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                    # Check if we should ignore exit events for this camera
                    if camera_id == IGNORE_EXIT_CAMERA_ID:
                        print(f"Process {os.getpid()} - Camera {camera_id} - Vehicle {track_id} exited but exit event ignored (camera configured to ignore exits)")
                        track_state['prev_centroid'] = curr_centroid
                        continue
                   
                    state['exit_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "exit"
                    print(f"Process {os.getpid()} - Camera {camera_id} - Vehicle {track_id} exited. Total exiting: {state['exit_count']}")
                    
                    # Handle screenshot and logging based on camera type
                    if is_entry_exit_dedup_camera:
                        # For entry-exit deduplication camera, always save screenshot and log since we already checked deduplication at function start
                        filename = save_screenshot(screenshot_frame, camera_id, timestamp, "exit_")
                        shared_state["last_screenshot_timestamps"]["last_any_event"] = timestamp
                        print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for exit event (entry-exit dedup camera)")
                    elif is_special_camera:
                        # For special camera, check 20-second window
                        last_timestamp = shared_state["last_screenshot_timestamps"].get("special_camera_event")
                        if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SPECIAL_CAMERA_DEDUP_WINDOW:
                            filename = save_screenshot(screenshot_frame, camera_id, timestamp, "exit_")
                            shared_state["last_screenshot_timestamps"]["special_camera_event"] = timestamp
                            print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for exit event (special camera)")
                        else:
                            print(f"Process {os.getpid()} - Camera {camera_id} - Skipped screenshot and log for exit event (within {SPECIAL_CAMERA_DEDUP_WINDOW}s for special camera)")
                            event_type = None  # Prevent log creation
                    else:
                        # For normal cameras, use original logic
                        last_timestamp = shared_state["last_screenshot_timestamps"].get("vehicle_exit")
                        if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SCREENSHOT_TIME_WINDOW:
                            filename = save_screenshot(screenshot_frame, camera_id, timestamp, "exit_")
                            shared_state["last_screenshot_timestamps"]["vehicle_exit"] = timestamp
                            shared_state["last_screenshot_timestamps"]["global_event"] = timestamp
                            print(f"Process {os.getpid()} - Camera {camera_id} - Saved screenshot from earliest buffer frame and created log for exit event")
                        else:
                            print(f"Process {os.getpid()} - Camera {camera_id} - Skipped screenshot and log for exit event (within {SCREENSHOT_TIME_WINDOW}s)")
                            event_type = None  # Prevent log creation
           
            if event_type:
                log_entry = {
                    "modelName": MODEL_CONFIG["1"]["name"],
                    "logData": {
                        "time": timestamp.strftime("%d %b %Y %H:%M:%S"),
                        "eventType": event_type,
                        "screenShotPath": filename,
                        "cameraId": camera_id,
                        "location": camera_info.get("location", "UNKNOWN"),
                        "entryCount": state["enter_count"],
                        "exitCount": state["exit_count"]
                    }
                }
           
            if abs(curr_dist) > CROSSING_TOLERANCE * 2:
                track_state['has_crossed'] = False
           
            track_state['prev_centroid'] = curr_centroid
       
        return log_entry
    finally:
        gc.collect()

def append_to_json_log(log_entry, camera_id):
    """Send log entry to the output API endpoint with retries."""
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(OUTPUT_API_URL, json=log_entry, timeout=5)
            if response.status_code == 200:
                print(f"Process {os.getpid()} - Camera {camera_id} - Successfully sent log entry to {OUTPUT_API_URL}: {log_entry}")
                return True
            else:
                print(f"Process {os.getpid()} - Camera {camera_id} - Failed to send log entry: Status {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            print(f"Process {os.getpid()} - Camera {camera_id} - Failed to send log entry to {OUTPUT_API_URL}: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2
    print(f"Process {os.getpid()} - Camera {camera_id} - Failed to send log entry after {max_retries} attempts")
    return False

def select_roi(camera_id, first_frame, width, height, state):
    """Load ROI configuration for a camera from saved JSON file."""
    if first_frame is None or first_frame.size == 0 or first_frame.shape[0] == 0 or first_frame.shape[1] == 0:
        error_msg = f"Invalid first frame for camera {camera_id}"
        print(f"Process {os.getpid()} - {error_msg}")
        report_error(camera_id, error_msg)
        return False

    saved_roi = load_roi_config(camera_id)
   

    if saved_roi:
        state["roi_points"] = saved_roi
        state["roi_selected"] = True
        print(f"Process {os.getpid()} - Camera {camera_id} - Loaded saved ROI: {saved_roi}")
        return True
    else:
        error_msg = f"No saved ROI configuration found for camera {camera_id}. Please provide a valid ROI configuration file."
        print(f"Process {os.getpid()} - {error_msg}")
        report_error(camera_id, error_msg)
        return False

def process_camera_worker(camera, model_ids):
    """Process video stream for a single camera for specified model IDs in a separate process."""
    camera_id = camera["cameraId"]
    video_path = camera["rtspUrl"]
    
    # Create local stop event for this process
    local_stop_event = threading.Event()
    
    # Setup signal handlers for this process
    def local_signal_handler(signum, frame):
        print(f"Process {os.getpid()} - Camera {camera_id} - Received signal {signum}. Stopping processing.")
        local_stop_event.set()
    
    signal.signal(signal.SIGINT, local_signal_handler)
    signal.signal(signal.SIGTERM, local_signal_handler)
    
    # Setup process-specific logging
    process_logger = logging.getLogger(f"camera_{camera_id}_process_{os.getpid()}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    process_logger.addHandler(handler)
    process_logger.setLevel(logging.INFO)
    
    print(f"Process {os.getpid()} - Processing camera {camera_id} with video URL: {video_path} for modelIds {model_ids}")
    
    # Debug: Print the camera information
    print(f"Process {os.getpid()} - Camera {camera_id} - Full camera info: {camera}")
    print(f"Process {os.getpid()} - Camera {camera_id} - Video path type: {type(video_path)}, Value: '{video_path}'")
    
    # Load YOLO model in this process
    vehicle_model, device = load_yolo_model()
    if vehicle_model is None:
        error_msg = f"Failed to load YOLO model in process {os.getpid()}"
        print(f"Process {os.getpid()} - {error_msg}")
        report_error(camera_id, error_msg)
        return
    
    # Initialize state for each model and shared state (local to this process)
    camera_state = {
        "shared": {
            "cap": None,
            "width": None,
            "height": None,
            "roi_points": [],
            "roi_selected": False,
            "line_params": None,
            "last_screenshot_timestamps": {
                "vehicle_enter": None,
                "vehicle_exit": None,
                "global_event": None,
                "special_camera_event": None,  # For special camera 20-second deduplication
                "last_any_event": None  # For entry-exit deduplication camera
            },
            "frame_buffer": deque(maxlen=FRAME_BUFFER_SIZE)  # Initialize frame buffer
        }
    }
    for model_id in model_ids:
        camera_state[model_id] = initialize_camera_state(model_id, camera_id)
   
    state = camera_state["shared"]
   
    # Initialize DeepSORT
    deepsort = DeepSort(max_age=10, nn_budget=100, override_track_class=None)
   
    # Load video
    cap = None
    try:
        if Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
       
        if not cap.isOpened():
            error_msg = f"Failed to open stream for camera {camera_id}"
            print(f"Process {os.getpid()} - {error_msg}")
            report_error(camera_id, error_msg)
            return
       
        state["cap"] = cap
       
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Process {os.getpid()} - Camera {camera_id} - Width: {width}, Height: {height}, FPS: {fps}")
       
        # Resize frame if resolution is low
        if width < 640 or height < 480:
            scale_factor = 2
            width = int(width * scale_factor)
            height = int(height * scale_factor)
       
        state["width"] = width
        state["height"] = height
       
        # Read first frame for ROI selection
        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            error_msg = f"Failed to read first frame for camera {camera_id}"
            print(f"Process {os.getpid()} - {error_msg}")
            report_error(camera_id, error_msg)
            return
       
        first_frame = cv2.resize(first_frame, (width, height))
        print(f"Process {os.getpid()} - Camera {camera_id} - First frame shape: {first_frame.shape}")
       
        # Select ROI
        if not select_roi(camera_id, first_frame, width, height, state):
            error_msg = f"Failed to select ROI for camera {camera_id}"
            print(f"Process {os.getpid()} - {error_msg}")
            report_error(camera_id, error_msg)
            return
       
        # Define ROI line
        x1, y1 = state["roi_points"][0]
        x2, y2 = state["roi_points"][1]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        state["line_params"] = (a, b, c)
       
        # Process video frames
        frame_count = 0
        while cap.isOpened() and not local_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                error_msg = f"End of video or failed to read frame for camera {camera_id}"
                print(f"Process {os.getpid()} - {error_msg}")
                report_error(camera_id, error_msg)
                break
           
            frame = cv2.resize(frame, (width, height))
           
            # Store frame in buffer
            state["frame_buffer"].append(frame.copy())
           
            # Skip frames to reduce CPU usage
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                del frame
                continue
           
            # Determine confidence threshold based on time
            current_hour = datetime.now().hour
            conf_threshold = 0.4 if 19 <= current_hour < 6 else 0.5
            print(f"Process {os.getpid()} - Camera {camera_id} - Using confidence threshold: {conf_threshold}")
           
            # Process each model
            for model_id in model_ids:
                model_state = camera_state[model_id]
               
                # Perform vehicle detection on full frame
                try:
                    # Run YOLO inference on GPU
                    results = vehicle_model(frame, classes=MODEL_CONFIG[model_id]["classes"], device=device)
                    detections = results[0].boxes.data.cpu().numpy()
                    
                    print(f"Process {os.getpid()} - Camera {camera_id} - Model {model_id} - Frame processed: {len(detections)} vehicle detections")
                except Exception as e:
                    error_msg = f"Vehicle detection failed for camera {camera_id}, model {model_id}: {e}"
                    print(f"Process {os.getpid()} - {error_msg}")
                    report_error(camera_id, error_msg)
                    continue
               
                # Prepare detections for DeepSORT
                deepsort_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf < conf_threshold:
                        continue
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                    deepsort_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
               
                # Update DeepSORT tracker
                tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
                
                print(f"Process {os.getpid()} - Camera {camera_id} - Model {model_id} - DeepSORT tracks: {len(tracks)}")
               
                # Determine crossing based on model ID
                log_entry = None
                if model_id == "1":
                    log_entry = determine_crossing_vehicle(
                        tracks, a, b, c, frame, camera_id, camera, model_state, state
                    )
               
                if log_entry:
                    # Offload logging to a separate thread to reduce delay
                    threading.Thread(target=append_to_json_log, args=(log_entry, camera_id), daemon=True).start()
                    print(f"Process {os.getpid()} - Camera {camera_id} - Model {model_id} - Dispatched log entry to separate thread: {log_entry}")
               
                # Free memory
                del results, detections, deepsort_detections, tracks
                # Clear GPU cache if using CUDA
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
           
            del frame
            # Clear GPU cache periodically if using CUDA
            if device == 'cuda' and frame_count % 50 == 0:  # Every 50 processed frames
                torch.cuda.empty_cache()
            gc.collect()
   
    except Exception as e:
        error_msg = f"Unexpected error in camera {camera_id}: {e}"
        print(f"Process {os.getpid()} - {error_msg}")
        report_error(camera_id, error_msg)
    finally:
        if cap is not None:
            cap.release()
        # Clear GPU cache when camera processing ends
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print(f"Process {os.getpid()} - Camera {camera_id} - Processing stopped")

def process_camera(camera, model_ids):
    """Legacy function for backward compatibility - redirects to worker function."""
    return process_camera_worker(camera, model_ids)

def fetch_cameras():
    """Fetch camera data from the input API with retries."""
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            print(f"Main Process - Fetching camera data from {INPUT_API_URL} (Attempt {attempt + 1}/{max_retries})")
            response = requests.get(INPUT_API_URL, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch camera data: Status {response.status_code}, Response: {response.text}")
            cameras = response.json()
            print(f"Main Process - Received camera data: {len(cameras)} cameras")
            return cameras
        except (requests.RequestException, RuntimeError) as e:
            error_msg = f"Failed to fetch camera data: {e}"
            print(f"Main Process - {error_msg}")
            report_error("system", error_msg)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Main Process - Max retries reached. Will retry on next fetch cycle.")
                return []
    return []

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global stop_event
    print(f"Main Process - Received signal {signum}. Stopping all processing.")
    if stop_event is not None:
        stop_event.set()

def main():
    """Run processing continuously, fetching camera data every hour or on error confirmation."""
    global stop_event, fetch_trigger
    
    # Set multiprocessing start method to 'spawn' for better isolation
    mp.set_start_method('spawn', force=True)
    
    # Initialize multiprocessing objects after setting spawn context (no manager needed)
    stop_event = mp.Event()
    fetch_trigger = mp.Event()  # Event to trigger immediate fetch after error confirmation
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Use multiprocessing for camera processing
    active_cameras = set()
    active_processes = {}  # {camera_id: process}
   
    try:
        while not stop_event.is_set():
            # Fetch camera data
            if fetch_trigger.is_set():
                print("Main Process - Immediate fetch triggered due to error confirmation")
                fetch_trigger.clear()
            
            cameras = fetch_cameras()
            if not cameras:
                print("Main Process - No cameras received, waiting before retry...")
                time.sleep(10)  # Short wait before retry
                continue
           
            # Group cameras by camera ID and collect model IDs
            camera_models = {}
            for cam in cameras:
                camera_id = cam["cameraId"]
                if camera_id not in camera_models:
                    camera_models[camera_id] = {"camera": cam, "model_ids": []}
                for model in cam.get("aiModels", []):
                    model_id = model["modelId"]
                    if model_id in MODEL_CONFIG and model_id not in camera_models[camera_id]["model_ids"]:
                        camera_models[camera_id]["model_ids"].append(model_id)
           
            # Log camera and model information
            for camera_id, info in camera_models.items():
                print(f"Main Process - Camera {camera_id} has modelIds: {info['model_ids']}")
           
            # Clean up inactive cameras
            new_camera_ids = set(camera_models.keys())
            cameras_to_stop = active_cameras - new_camera_ids
            
            for cam_id in cameras_to_stop:
                if cam_id in active_processes:
                    process = active_processes[cam_id]
                    if process.is_alive():
                        print(f"Main Process - Terminating process for camera {cam_id}")
                        process.terminate()
                        process.join(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                        if process.is_alive():
                            print(f"Main Process - Force killing process for camera {cam_id}")
                            process.kill()
                    del active_processes[cam_id]
                    print(f"Main Process - Stopped processing for camera {cam_id} (no longer in input)")
           
            active_cameras = new_camera_ids
            
            # Start new camera processes (with limit check)
            for camera_id, info in camera_models.items():
                # Check if we've reached the maximum number of processes
                if len(active_processes) >= MAX_PROCESSES and camera_id not in active_processes:
                    print(f"Main Process - Maximum processes ({MAX_PROCESSES}) reached. Camera {camera_id} will wait.")
                    continue
                    
                if camera_id not in active_processes:
                    # Create new process for this camera
                    process = mp.Process(
                        target=process_camera_worker,
                        args=(info["camera"], info["model_ids"]),
                        name=f"camera_{camera_id}_process"
                    )
                    process.start()
                    active_processes[camera_id] = process
                    print(f"Main Process - Started processing for camera {camera_id} with modelIds {info['model_ids']} in process {process.pid}")
                else:
                    # Check if existing process is still alive
                    process = active_processes[camera_id]
                    if not process.is_alive():
                        print(f"Main Process - Process for camera {camera_id} died, restarting...")
                        # Clean up dead process
                        process.join()
                        del active_processes[camera_id]
                        
                        # Start new process
                        new_process = mp.Process(
                            target=process_camera_worker,
                            args=(info["camera"], info["model_ids"]),
                            name=f"camera_{camera_id}_process"
                        )
                        new_process.start()
                        active_processes[camera_id] = new_process
                        print(f"Main Process - Restarted processing for camera {camera_id} in process {new_process.pid}")
           
            gc.collect()
            
            # Wait before next fetch cycle
            print(f"Main Process - Waiting {FETCH_INTERVAL} seconds before next fetch cycle...")
            time.sleep(FETCH_INTERVAL)
   
    except KeyboardInterrupt:
        print("Main Process - Received shutdown signal. Stopping all processing.")
        stop_event.set()
    except Exception as e:
        print(f"Main Process - Critical error in main loop: {e}")
        report_error("system", f"Critical error in main loop: {e}")
    finally:
        print("Main Process - Shutting down all camera processes...")
        stop_event.set()
        
        # Terminate all active processes
        for cam_id, process in active_processes.items():
            if process.is_alive():
                print(f"Main Process - Terminating process for camera {cam_id}")
                process.terminate()
                process.join(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                if process.is_alive():
                    print(f"Main Process - Force killing process for camera {cam_id}")
                    process.kill()
        
        # Clear shared state
        active_processes.clear()
        
        gc.collect()
        print("Main Process - Application shutdown complete.")

if __name__ == "__main__":
    main()