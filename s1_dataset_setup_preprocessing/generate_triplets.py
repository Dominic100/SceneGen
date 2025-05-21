# generate_triplets.py
import os
import json

frames_dir = r"C:\Aneesh\EDI VI\data\frames"
audio_dir = r"C:\Aneesh\EDI VI\data\audio"

with open(r"C:\Aneesh\EDI VI\data\processed\cleaned_captions.json", "r") as f:
    cleaned_captions = json.load(f)

triplets = []

for entry in cleaned_captions:
    video_id = entry["video_id"]
    start_time = entry["start_time"]
    caption = entry["caption"]

    # Frame file
    frame_filename = f"{video_id}_{int(start_time*1000)}ms.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)

    # Audio file (try to find corresponding audio clip with matching start_time)
    # Note: audio clips have end times, but cleaned_captions only has start_time,
    # so just pick the first audio clip starting at this start_time

    audio_candidates = [f for f in os.listdir(audio_dir) if f.startswith(f"{video_id}_{int(start_time*1000)}ms")]
    if not audio_candidates:
        # No audio clip found for this caption; skip or set None
        continue
    audio_path = os.path.join(audio_dir, audio_candidates[0])

    # Check if files exist
    if not os.path.exists(frame_path) or not os.path.exists(audio_path):
        continue

    triplets.append({
        "frame_path": frame_path,
        "audio_path": audio_path,
        "caption": caption
    })

os.makedirs(r"C:\Aneesh\EDI VI\data\processed", exist_ok=True)
with open(r"C:\Aneesh\EDI VI\data\processed\triplets.json", "w") as f:
    json.dump(triplets, f, indent=2)

print(f"Generated {len(triplets)} triplets in data/processed/triplets.json")
