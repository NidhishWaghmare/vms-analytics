#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import logging
import sys
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import os
import logging.handlers
import multiprocessing as mp
import requests
import time
import gc
import threading
import boto3
from botocore.client import Config
import signal
import uuid
import torch
from collections import deque

# Configuration
MODEL_PATH = "models/yolov8x.pt"  # Path to vehicle detection YOLO model
CLASSES = [5, 7]  # Classes: 5=bus, 7=truck
FRAME_SKIP = 5  # Process every 5th frame to reduce CPU usage
SCREENSHOT_TIME_WINDOW = 15  # 15 seconds to ignore duplicate same events
OUTPUT_DIR = Path("data/screenshots")
JSON_DIR = Path("data/logs")
ROI_DIR = Path("data/roi_configs")
VIDEO_DIR = Path("videos")
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 5  # Keep 5 log file backups

# MP4 Recording Configuration
RECORDINGS_BASE_PATH = Path("/home/ubantu/recordings")
CAMERA_IP_TO_UUID = {
    "192.168.1.104": "uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d",
    "192.168.1.109": "uuid:afd04419-897a-4f6c-8108-7137d9a2c1b8", 
    "192.168.1.249": "uuid:0f1d2d49-86cf-49d9-98fc-34750541b05d"
}

GLOBAL_EVENT_TIME_WINDOW = 1  # Seconds to deduplicate any event type
SPECIAL_CAMERA_DEDUP_WINDOW = 20  # Seconds for special camera deduplication
ENTRY_EXIT_DEDUP_WINDOW = 15  # Seconds to treat entry and exit as single detection
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

# Digital Ocean Spaces Configuration
DO_SPACE_ACCESS_KEY = 'DO801UYGLUGLVCDQFYNM'
DO_SPACE_SECRET_KEY = 'fBDdr0Cp5NmbkSkD0jeRgE+oIaOZcOdSfzOautQGnL4'
DO_SPACE_REGION = 'blr1'  # Region for the provided endpoint
DO_SPACE_NAME = 'vigilscreenshots'  # Space name from the endpoint
DO_SPACE_ENDPOINT = 'https://blr1.digitaloceanspaces.com'  # Endpoint for blr1 region
DO_SPACE_FOLDER = "screenshots"  # Folder in the Space to store screenshots

# API Configuration
DEFAULT_API_IP = "192.168.1.38:8001"  # API IP
OUTPUT_API_ENDPOINT = "/api/v1/aiAnalytics/sendAnalyticsJson"
INPUT_API_ENDPOINT = "/api/v1/aiAnalytics/getCamerasForAnalytics"
ERROR_API_ENDPOINT = "/api/v1/aiAnalytics/reportError"
API_IP = os.environ.get('API_IP', DEFAULT_API_IP)
OUTPUT_API_URL = f"http://{API_IP}{OUTPUT_API_ENDPOINT}"
INPUT_API_URL = f"http://{API_IP}{INPUT_API_ENDPOINT}"
ERROR_API_URL = f"http://{API_IP}{ERROR_API_ENDPOINT}"

# Camera ID to IP mapping
CAMERA_IP_MAPPING = {
    "0f1d2d49-86cf-49d9-98fc-34750541b05d": "192.168.1.249",
    "53b3850d-e0ef-4668-9fb5-12c980aac83d": "192.168.1.104",
    "afd04419-897a-4f6c-8108-7137d9a2c1b8": "192.168.1.109"
}

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available, falling back to CPU")

# Initialize boto3 client for Digital Ocean Spaces
s3_client = boto3.client(
    's3',
    region_name=DO_SPACE_REGION,
    endpoint_url=DO_SPACE_ENDPOINT,
    aws_access_key_id=DO_SPACE_ACCESS_KEY,
    aws_secret_access_key=DO_SPACE_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
ROI_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

# Setup logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            JSON_DIR / 'analytics.log',
            maxBytes=LOG_MAX_SIZE,
            backupCount=LOG_BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def do_intersect(p1, q1, p2, q2):
    """Check if line segment p1-q1 intersects with p2-q2."""
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        if (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1])):
            return True
        return False

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

def load_roi(camera_id):
    """Load ROI coordinates from JSON file based on camera ID."""
    roi_file = ROI_DIR / f"{camera_id}_roi.json"
    try:
        with open(roi_file, 'r') as f:
            data = json.load(f)
            points = data.get("roi_points", [])
            if len(points) != 2:
                raise ValueError("ROI JSON must contain exactly two points")
            return [(int(p[0]), int(p[1])) for p in points]
    except Exception as e:
        logger.error(f"Failed to load ROI for camera {camera_id}: {e}")
        sys.exit(1)

def report_error(camera_id, error_message):
    """Send error details to the error API endpoint."""
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
            logger.info(f"Successfully reported error for camera {camera_id}")
        else:
            logger.error(f"Failed to report error: Status {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to report error to {ERROR_API_URL}: {e}")
    finally:
        gc.collect()

def save_screenshot(frame, camera_id, timestamp, prefix):
    """Save a full frame screenshot to Digital Ocean Space and locally for backup, return cloud filename."""
    try:
        timestamp_str = timestamp.strftime("%d %b %Y %H:%M:%S").replace(" ", "_").replace(":", "_")
        local_filename = f"{prefix}{camera_id}_{timestamp_str}.jpg"
        local_filepath = OUTPUT_DIR / local_filename
        cv2.imwrite(str(local_filepath), frame)
        logger.info(f"Saved local backup screenshot: {local_filepath}")
        
        cloud_filename = f"{DO_SPACE_FOLDER}/{local_filename}"
        
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        file_bytes = buffer.tobytes()
        
        # Upload to Digital Ocean Space
        s3_client.put_object(
            Bucket=DO_SPACE_NAME,
            Key=cloud_filename,
            Body=file_bytes,
            ContentType='image/jpeg',
            ACL='public-read'  # Adjust ACL as needed
        )
        
        logger.info(f"Uploaded screenshot to Digital Ocean Space: {cloud_filename}")
        return cloud_filename
    except Exception as e:
        logger.error(f"Failed to save/upload screenshot: {e}")
        return None
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
                logger.info(f"Successfully sent log entry to {OUTPUT_API_URL}: {log_entry}")
                return True
            else:
                logger.error(f"Failed to send log entry: Status {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            logger.error(f"Failed to send log entry to {OUTPUT_API_URL}: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2
    logger.error(f"Failed to send log entry after {max_retries} attempts")
    return False

def determine_crossing_vehicle(tracks, a, b, c, frame, state, video_name, camera_id):
    """Determine if vehicle centroids cross the ROI line and classify as Entering/Exiting."""
    try:
        timestamp = datetime.now()
        
        # Check if this is one of the special cameras that requires 20-second deduplication
        is_special_camera = camera_id in SPECIAL_CAMERA_IDS
        
        # Check if this is the camera that needs entry-exit deduplication
        is_entry_exit_dedup_camera = camera_id == ENTRY_EXIT_DEDUP_CAMERA_ID
        
        # For entry-exit deduplication camera, check if any event happened within 10 seconds
        if is_entry_exit_dedup_camera:
            if (state.get("last_any_event_timestamp") is not None and 
                (timestamp - state["last_any_event_timestamp"]).total_seconds() <= ENTRY_EXIT_DEDUP_WINDOW):
                logger.debug(f"Skipped event due to entry-exit deduplication window (within {ENTRY_EXIT_DEDUP_WINDOW}s)")
                return None
        elif is_special_camera:
            if (state.get("last_special_event_timestamp") is not None and 
                (timestamp - state["last_special_event_timestamp"]).total_seconds() <= SPECIAL_CAMERA_DEDUP_WINDOW):
                logger.debug(f"Skipped event due to special camera deduplication window (within {SPECIAL_CAMERA_DEDUP_WINDOW}s)")
                return None
        else:
            if (state.get("last_event_timestamp") is not None and 
                (timestamp - state["last_event_timestamp"]).total_seconds() <= SCREENSHOT_TIME_WINDOW):
                logger.debug(f"Skipped duplicate event (within {SCREENSHOT_TIME_WINDOW}s)")
                return None
        
        log_entry = None
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            centroid = calculate_centroid(bbox)
            
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
            if not track_state['has_crossed']:
                intersects = do_intersect(prev_centroid, curr_centroid, state['roi_points'][0], state['roi_points'][1])
                if prev_dist < 0 and curr_dist >= 0 and intersects:
                    # Check if this camera should ignore entry events
                    if camera_id == IGNORE_ENTRY_CAMERA_ID:
                        logger.debug(f"Ignoring entry event for camera {camera_id}")
                        track_state['prev_centroid'] = curr_centroid
                        continue
                        
                    state['enter_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "enter"
                    logger.info(f"Vehicle {track_id} entered. Total entering: {state['enter_count']}")
                    filename = save_screenshot(frame, camera_id, timestamp, "enter_")
                    
                    # Update timestamps based on camera type
                    if is_entry_exit_dedup_camera:
                        state["last_any_event_timestamp"] = timestamp
                    elif is_special_camera:
                        state["last_special_event_timestamp"] = timestamp
                    else:
                        state["last_event_timestamp"] = timestamp
                        
                elif prev_dist > 0 and curr_dist <= 0 and intersects:
                    # Check if this camera should ignore exit events
                    if camera_id == IGNORE_EXIT_CAMERA_ID:
                        logger.debug(f"Ignoring exit event for camera {camera_id}")
                        track_state['prev_centroid'] = curr_centroid
                        continue
                        
                    state['exit_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "exit"
                    logger.info(f"Vehicle {track_id} exited. Total exiting: {state['exit_count']}")
                    filename = save_screenshot(frame, camera_id, timestamp, "exit_")
                    
                    # Update timestamps based on camera type
                    if is_entry_exit_dedup_camera:
                        state["last_any_event_timestamp"] = timestamp
                    elif is_special_camera:
                        state["last_special_event_timestamp"] = timestamp
                    else:
                        state["last_event_timestamp"] = timestamp
            
            if event_type:
                # Extract IP from camera_id to create location in format: location_foldername(ip)
                camera_ip = None
                for ip, uuid in CAMERA_IP_TO_UUID.items():
                    if uuid == camera_id:
                        camera_ip = ip
                        break
                
                location = f"location_{camera_ip}" if camera_ip else "test_location"
                
                log_entry = {
                    "modelName": "vehicleInOut",
                    "logData": {
                        "time": timestamp.strftime("%d %b %Y %H:%M:%S"),
                        "eventType": event_type,
                        "screenShotPath": filename,
                        "cameraId": camera_id,
                        "location": location,
                        "entryCount": state["enter_count"],
                        "exitCount": state["exit_count"]
                    }
                }
                # Offload logging to a separate thread to reduce delay
                threading.Thread(target=append_to_json_log, args=(log_entry, camera_id), daemon=True).start()
            
            if abs(curr_dist) > 20:
                track_state['has_crossed'] = False
            
            track_state['prev_centroid'] = curr_centroid
        
        return log_entry
    except Exception as e:
        logger.error(f"Error in determine_crossing_vehicle: {e}")
        return None
    finally:
        gc.collect()

def process_video(ip_folder_path, camera_id):
    """Process video chunks from IP folder for a camera."""
    ip_folder = Path(ip_folder_path)
    logger.info(f"Processing camera {camera_id} with chunks from folder: {ip_folder}")

    # Load YOLO model
    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        # Move model to GPU if available
        if device == 'cuda':
            model.to(device)
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return

    # Initialize DeepSort
    deepsort = DeepSort(max_age=10, nn_budget=100, override_track_class=None)

    # Load ROI points
    roi_points = load_roi(camera_id)
    if len(roi_points) != 2:
        logger.error(f"Invalid ROI points for camera {camera_id}")
        return

    # Initialize state
    state = {
        "enter_count": 0,
        "exit_count": 0,
        "track_states": {},
        "last_event_timestamp": None,
        "last_special_event_timestamp": None,
        "last_any_event_timestamp": None,
        "roi_points": roi_points,
        "frame_buffer": deque(maxlen=FRAME_BUFFER_SIZE)
    }

    # Calculate line parameters
    x1, y1 = roi_points[0]
    x2, y2 = roi_points[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = np.sqrt(a**2 + b**2)
    if norm == 0:
        logger.error("Invalid ROI line (zero length)")
        return
    logger.info(f"ROI line equation: {a}x + {b}y + {c} = 0")

    # Track which chunk we're currently processing
    current_chunk_index = 0
    width = None
    height = None
    
    logger.info(f"Starting continuous chunk processing from chunk_{current_chunk_index:03d}.mp4")
    
    try:
        # Continuous processing loop - keep running until stopped
        while True:
            # Look for the next chunk file
            chunk_filename = f"chunk_{current_chunk_index:03d}.mp4"
            chunk_file = ip_folder / chunk_filename
            
            # Wait for the chunk file to exist
            if not chunk_file.exists():
                logger.info(f"Waiting for {chunk_filename} to be created...")
                time.sleep(5)  # Wait 5 seconds before checking again
                continue
                
            logger.info(f"Found {chunk_filename}, starting processing...")
            
            cap = None
            try:
                cap = cv2.VideoCapture(str(chunk_file))
                
                if not cap.isOpened():
                    logger.error(f"Failed to open chunk file {chunk_file}")
                    current_chunk_index += 1
                    continue
                
                # Get video properties (only from first video)
                if width is None or height is None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    logger.info(f"Camera {camera_id} - Width: {width}, Height: {height}, FPS: {fps}")
                    
                    # Resize frame if resolution is low
                    if width < 640 or height < 480:
                        scale_factor = 2
                        width = int(width * scale_factor)
                        height = int(height * scale_factor)
                        logger.info(f"Resizing to: {width}x{height}")

                # Process video frames from this chunk
                frame_count = 0
                last_tracks = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        logger.info(f"Finished processing {chunk_filename}")
                        break

                    frame = cv2.resize(frame, (width, height))
                    
                    # Add frame to buffer
                    state["frame_buffer"].append(frame.copy())
                    tracks = last_tracks
                    
                    if frame_count % FRAME_SKIP == 0:
                        try:
                            current_hour = datetime.now().hour
                            conf_threshold = 0.4 if 19 <= current_hour or current_hour < 6 else 0.5
                            results = model(frame, classes=CLASSES, conf=conf_threshold, verbose=False)
                            detections = results[0].boxes.data.cpu().numpy() if len(results[0].boxes) > 0 else []
                            
                            if frame_count % (FRAME_SKIP * 10) == 0:
                                logger.info(f"Frame {frame_count}: {len(detections)} detections")
                                
                        except Exception as e:
                            logger.error(f"Vehicle detection failed on frame {frame_count}: {e}")
                            frame_count += 1
                            continue

                        deepsort_detections = []
                        for det in detections:
                            x1, y1, x2, y2, conf, cls = det
                            if conf < conf_threshold:
                                continue
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            deepsort_detections.append((bbox, conf, cls))

                        try:
                            tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
                            last_tracks = tracks
                        except Exception as e:
                            logger.error(f"DeepSort tracking failed on frame {frame_count}: {e}")
                            frame_count += 1
                            continue

                        # Check for vehicle crossings
                        if tracks:
                            log_entry = determine_crossing_vehicle(tracks, a, b, c, frame, state, chunk_filename, camera_id)

                    frame_count += 1

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_filename}: {e}")
            finally:
                if cap is not None:
                    cap.release()
                gc.collect()
                
            # Move to next chunk
            current_chunk_index += 1
            
    except KeyboardInterrupt:
        logger.info(f"Processing interrupted for camera {camera_id}")
    except Exception as e:
        logger.error(f"Unexpected error in camera {camera_id}: {e}")
        report_error(camera_id, str(e))
    finally:
        # Clear GPU cache when camera processing ends
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Camera {camera_id} processing stopped")


def discover_mp4_files():
    """Discover camera folders and create camera data structure for initial setup."""
    cameras = []
    
    for camera_ip, camera_uuid in CAMERA_IP_TO_UUID.items():
        camera_folder = RECORDINGS_BASE_PATH / camera_ip
        if not camera_folder.exists():
            logger.warning(f"Camera folder {camera_folder} does not exist, skipping camera {camera_ip}")
            continue
            
        logger.info(f"Setting up camera {camera_ip} (UUID: {camera_uuid}) to monitor folder: {camera_folder}")
        
        # Create camera data structure matching the API format
        camera_data = {
            "cameraId": camera_uuid,
            "rtspUrl": str(camera_folder),
            "location": f"Location_{camera_ip}",
            "aiModels": [{"modelId": "1"}]
        }
        cameras.append(camera_data)
        
    return cameras


def process_camera_worker(camera, model_ids):
    """Process video stream for a single camera for specified model IDs in a separate process."""
    camera_id = camera["cameraId"]
    camera_folder = Path(camera["rtspUrl"])
    
    logger.info(f"Process {os.getpid()} - Processing camera {camera_id} with MP4 folder: {camera_folder} for modelIds {model_ids}")
    
    try:
        process_video(str(camera_folder), camera_id)
    except Exception as e:
        error_msg = f"Unexpected error in camera {camera_id}: {e}"
        logger.error(f"Process {os.getpid()} - {error_msg}")
        report_error(camera_id, error_msg)
    finally:
        # Clear GPU cache when camera processing ends
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Process {os.getpid()} - Camera {camera_id} - Processing stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}. Stopping all processing.")


def main():
    """Main function to start the VMS analytics service."""
    # Set multiprocessing start method to 'spawn' for better isolation
    mp.set_start_method('spawn', force=True)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize cameras once
    cameras = discover_mp4_files()
    
    if not cameras:
        logger.error("No cameras with MP4 files found!")
        return
    
    logger.info(f"Starting processing for {len(cameras)} cameras")
    
    # Start processes for all cameras and let them run continuously
    processes = []
    for camera in cameras:
        process = mp.Process(
            target=process_camera_worker, 
            args=(camera, ["1"]),
            name=f"camera_{camera['cameraId']}_process"
        )
        process.start()
        processes.append(process)
        logger.info(f"Started process for camera {camera['cameraId']} monitoring folder: {camera['rtspUrl']}")
    
    try:
        logger.info("All camera processes started. They will monitor and process chunks continuously.")
        logger.info("Press Ctrl+C to stop all processes")
        
        # Wait for all processes to complete (they run indefinitely)
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        logger.info("Stopping all processes...")
        for process in processes:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        logger.info("All processes stopped")


if __name__ == "__main__":
    main()