import json
import os
from collections import defaultdict

# ====== CONFIGURATION ======
INPUT_PATH = 'C:/Aneesh/EDI VI/data/processed/multimodal_triplets.json'
OUTPUT_PATH = 'C:/Aneesh/EDI VI/data/processed/scene_level_windows.json'
WINDOW_SIZE = 5
STRIDE = 1

# ====== LOAD TRIPLETS ======
with open(INPUT_PATH, 'r') as f:
    triplets = json.load(f)

# Group by video_id prefix (assumes full video ID is everything before first underscore)
grouped = defaultdict(list)
for entry in triplets:
    base_video_id = entry['video_id'].split('_')[0]
    grouped[base_video_id].append(entry)

# Sort by image_path (which contains timestamp — assumed to be increasing)
for video_id in grouped:
    grouped[video_id] = sorted(grouped[video_id], key=lambda x: x['image_path'])

# ====== BUILD TEMPORAL WINDOWS ======
scene_windows = []
scene_id = 0

for video_id, entries in grouped.items():
    num_segments = len(entries)
    for start in range(0, num_segments - WINDOW_SIZE + 1, STRIDE):
        window = entries[start:start + WINDOW_SIZE]
        scene_windows.append({
            'scene_id': f'{video_id}_scene_{scene_id}',
            'video_id': video_id,
            'entries': window
        })
        scene_id += 1

# ====== SAVE OUTPUT ======
with open(OUTPUT_PATH, 'w') as f:
    json.dump(scene_windows, f, indent=4)

print(f"✅ Created {len(scene_windows)} scene-level windows.")
print(f"📁 Saved to: {OUTPUT_PATH}")
