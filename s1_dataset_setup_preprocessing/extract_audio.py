import json
import os
import subprocess

# Load raw captions again to get start & end times
with open(r"C:\Aneesh\EDI VI\data\captions.json", "r") as f:
    raw_captions = json.load(f)

audio_dir = r"C:\Aneesh\EDI VI\data\audio"
os.makedirs(audio_dir, exist_ok=True)

for video_id, info in raw_captions.items():
    starts = info.get("start", [])
    ends = info.get("end", [])
    texts = info.get("text", [])

    if not (len(starts) == len(ends) == len(texts)):
        print(f"Skipping {video_id} due to mismatch in lengths")
        continue

    video_path = fr"C:\Aneesh\EDI VI\data\videos\{video_id}.mp4"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found, skipping.")
        continue

    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        caption = texts[i].strip()
        if not caption or end <= start:
            continue

        duration = end - start
        audio_filename = f"{video_id}_{int(start*1000)}ms_{int(end*1000)}ms.wav"
        output_path = os.path.join(audio_dir, audio_filename)
        
        # Use ffmpeg to extract the audio segment
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit encoding
            "-ar", "44100",  # 44.1kHz sample rate
            "-ac", "2",  # Stereo
            "-loglevel", "error",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio for {video_id} at {start}-{end}: {e}")

print("Audio extraction done.")