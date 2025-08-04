import cv2
import os
import math

def extract_frames(video_path, output_dir, min_frames=500):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_hours = 4  # 4 hours as specified
    
    # Calculate duration in seconds
    duration_seconds = duration_hours * 3600
    # Estimate total frames if not available (in case of corrupted metadata)
    if total_frames <= 0:
        total_frames = int(fps * duration_seconds)
    
    # Calculate the interval (n) to get at least min_frames
    n = max(1, math.floor(total_frames / min_frames))
    
    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Extracting every {n}th frame to get at least {min_frames} frames")
    
    count = 0
    frame_index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save every nth frame
        if frame_index % n == 0:
            output_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1
            
        frame_index += 1
        
        # Optional: Print progress
        if frame_index % 1000 == 0:
            print(f"Processed {frame_index} frames, saved {count} frames")
    
    cap.release()
    print(f"Total frames extracted: {count}")
    if count < min_frames:
        print(f"Warning: Extracted {count} frames, which is less than the requested {min_frames} frames.")

# Example usage
if __name__ == "__main__":
    video_path = "/home/tidalwave/recordings/192.168.1.4_b/chunk_002.mp4"  # Replace with your video file path
    output_dir = "/home/tidalwave/vms/Dataset"  # Directory where frames will be saved
    extract_frames(video_path, output_dir)
