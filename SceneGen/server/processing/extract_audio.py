import os
import subprocess
import json
import tempfile

def extract_random_audio_clips(audio_path, output_dir, num_clips=None, interval=2.0, max_duration=2.0, video_path=None):
    """
    Extract audio clips from an audio file at regular intervals
    
    Args:
        audio_path (str): Path to the audio file (can be None if using video_path)
        output_dir (str): Directory to save audio clips
        num_clips (int, optional): Number of clips to extract (if None, uses interval)
        interval (float): Time interval between clips in seconds
        max_duration (float): Maximum duration of each clip in seconds
        video_path (str, optional): Path to video file if audio_path is not provided
        
    Returns:
        list: List of dictionaries with audio clip information
    """
    # If audio_path is not provided, extract audio from video
    if not audio_path and video_path:
        print("No separate audio file provided. Extracting audio from video...")
        # Create temporary file for extracted audio
        temp_audio = os.path.join(tempfile.gettempdir(), 'temp_audio.wav')
        
        # Extract audio from video using ffmpeg
        extract_audio_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            temp_audio
        ]
        
        subprocess.run(extract_audio_cmd, capture_output=True)
        
        # Use the extracted audio file for further processing
        audio_path = temp_audio
        print(f"Audio extracted to temporary file: {audio_path}")
    
    # Get audio duration using ffprobe
    duration_cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'json', 
        audio_path
    ]
    
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration_data = json.loads(result.stdout)
    audio_duration = float(duration_data['format']['duration'])
    
    if not audio_duration:
        raise ValueError('Could not determine audio duration')
    
    # If num_clips not specified, calculate based on interval
    if num_clips is None:
        num_clips = max(1, int(audio_duration // interval))
    
    extracted_clips = []
    
    # Extract clips at regular intervals
    for i in range(num_clips):
        # Start time at regular intervals
        start_time = i * interval
        
        # Skip if beyond audio duration
        if start_time >= audio_duration:
            break
        
        # Calculate clip duration, ensuring it doesn't go beyond audio end
        clip_duration = min(max_duration, audio_duration - start_time)
        
        clip_id = f"audio_{str(i + 1).zfill(3)}"
        clip_path = os.path.join(output_dir, f"{clip_id}.wav")
        
        # Extract the clip using ffmpeg
        extract_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', audio_path,
            '-ss', str(start_time),
            '-t', str(clip_duration),
            '-acodec', 'pcm_s16le',
            '-ac', '2',
            '-ar', '44100',
            clip_path
        ]
        
        subprocess.run(extract_cmd, capture_output=True)
        
        extracted_clips.append({
            'id': clip_id,
            'path': clip_path,
            'startTime': float(start_time),
            'duration': float(clip_duration)
        })
    
    # Clean up temporary file if created
    if not audio_path and video_path and os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    # Sort clips by start time (already in order, but keeping for consistency)
    extracted_clips.sort(key=lambda x: x['startTime'])
    return extracted_clips