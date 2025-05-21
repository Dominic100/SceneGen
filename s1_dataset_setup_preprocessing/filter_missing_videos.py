# filter_missing_videos.py
import json
import os

# Paths
cleaned_captions_path = r"C:\Aneesh\EDI VI\data\processed\cleaned_captions.json"
videos_dir = r"C:\Aneesh\EDI VI\data\videos"

# Load cleaned captions
with open(cleaned_captions_path, "r") as f:
    cleaned_captions = json.load(f)

filtered_captions = []

for entry in cleaned_captions:
    video_id = entry["video_id"]
    video_path = os.path.join(videos_dir, f"{video_id}.mp4")
    if os.path.exists(video_path):
        filtered_captions.append(entry)
    else:
        print(f"Missing video for video_id: {video_id}, removing corresponding captions.")

# Overwrite cleaned_captions.json with filtered data
with open(cleaned_captions_path, "w") as f:
    json.dump(filtered_captions, f, indent=2)

print(f"Filtered cleaned_captions.json: kept {len(filtered_captions)} entries.")
