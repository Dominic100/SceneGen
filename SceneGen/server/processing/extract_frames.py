import cv2
import os
import numpy as np

def extract_random_frames(video_path, output_dir, num_frames=None, interval=2.0):
    """
    Extract frames from a video file at regular intervals
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save frames
        num_frames (int, optional): Number of frames to extract (if None, uses interval)
        interval (float): Time interval between frames in seconds
        
    Returns:
        list: List of dictionaries with frame information
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    if frame_count <= 0 or fps <= 0:
        raise ValueError('Invalid video file')
    
    # If num_frames not specified, calculate based on interval
    if num_frames is None:
        num_frames = max(1, int(duration // interval))
    
    extracted_frames = []
    
    # Extract frames at regular intervals
    for i in range(num_frames):
        # Calculate timestamp at regular intervals
        timestamp = i * interval
        
        # Skip if beyond video duration
        if timestamp >= duration:
            break
            
        # Calculate frame index
        frame_index = int(timestamp * fps)
        
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save frame
        frame_id = f"frame_{str(i + 1).zfill(3)}"
        frame_path = os.path.join(output_dir, f"{frame_id}.jpg")
        cv2.imwrite(frame_path, frame)
        
        extracted_frames.append({
            'id': frame_id,
            'path': frame_path,
            'timestamp': float(timestamp)
        })
    
    cap.release()
    return extracted_frames