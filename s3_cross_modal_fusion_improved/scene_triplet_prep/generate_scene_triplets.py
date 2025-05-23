import json
import random
from collections import defaultdict

# ====== CONFIG ======
INPUT_PATH = 'C:/Aneesh/EDI VI/data/processed/scene_level_windows.json'
OUTPUT_PATH = 'C:/Aneesh/EDI VI/data/processed/scene_triplets.json'
POS_RANGE = 2  # number of neighbors before/after to be considered positive
NEG_SAMPLE_ATTEMPTS = 10

# ====== LOAD SCENES ======
with open(INPUT_PATH, 'r') as f:
    scenes = json.load(f)

# Group by video_id
grouped = defaultdict(list)
for scene in scenes:
    grouped[scene['video_id']].append(scene)

# Sort each list by scene_id index
for video_id in grouped:
    grouped[video_id] = sorted(grouped[video_id], key=lambda x: int(x['scene_id'].split('_')[-1]))

triplets = []

# ====== GENERATE TRIPLETS ======
for video_id, scene_list in grouped.items():
    total = len(scene_list)
    for idx, anchor in enumerate(scene_list):
        # Pick positive
        pos_indices = list(range(max(0, idx - POS_RANGE), min(total, idx + POS_RANGE + 1)))
        pos_indices = [i for i in pos_indices if i != idx]  # exclude anchor
        if not pos_indices:
            continue
        pos_idx = random.choice(pos_indices)
        positive = scene_list[pos_idx]

        # Pick negative (from another video or distant scene)
        negative = None
        attempts = 0
        while not negative and attempts < NEG_SAMPLE_ATTEMPTS:
            other_video_id = random.choice(list(grouped.keys()))
            if other_video_id == video_id:
                # Choose a scene far apart
                distant = [s for s in grouped[video_id] if abs(int(s['scene_id'].split('_')[-1]) - idx) > POS_RANGE + 2]
                if distant:
                    negative = random.choice(distant)
            else:
                negative = random.choice(grouped[other_video_id])
            attempts += 1

        if negative:
            triplets.append({
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            })

# ====== SAVE ======
with open(OUTPUT_PATH, 'w') as f:
    json.dump(triplets, f, indent=4)

print(f"✅ Generated {len(triplets)} contrastive triplets.")
print(f"📁 Saved to: {OUTPUT_PATH}")
