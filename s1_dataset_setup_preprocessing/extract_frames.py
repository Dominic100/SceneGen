# extract_frames.py
import json
import os
import cv2

# Load cleaned captions list
with open(r"C:\Aneesh\EDI VI\data\processed\cleaned_captions.json", "r") as f:
    captions = json.load(f)

# Output directory for frames
frames_dir = r"C:\Aneesh\EDI VI\data\frames"
os.makedirs(frames_dir, exist_ok=True)

for entry in captions:
    video_id = entry["video_id"]
    start_time = entry["start_time"]
    video_path = fr"C:\Aneesh\EDI VI\data\videos\{video_id}.mp4"

    if not os.path.exists(video_path):
        print(f"Video {video_path} not found, skipping.")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: FPS is zero for {video_path}, skipping.")
        cap.release()
        continue

    frame_number = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {start_time}s in {video_path}, skipping.")
        cap.release()
        continue

    # Save frame as JPEG
    frame_filename = f"{video_id}_{int(start_time*1000)}ms.jpg"
    cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)
    cap.release()

print("Frame extraction done.")
