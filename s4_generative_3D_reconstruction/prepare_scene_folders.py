import os
import json
import shutil
from tqdm import tqdm

# Config - update these paths as needed
TRIPLETS_PATH = r"C:\Aneesh\EDI VI\data\processed\scene_triplets.json"
MASTER_FRAMES_DIR = r"C:\Aneesh\EDI VI\data\frames"  # All your frames here
OUTPUT_BASE_DIR = r"C:\Aneesh\EDI VI\data\scene_frames"  # Where scene folders will be created

# Supported image extensions - update if needed
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def find_frame_file(frame_name):
    """Finds the full path of the frame image file in MASTER_FRAMES_DIR"""
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(MASTER_FRAMES_DIR, frame_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None

def prepare_scene_folders():
    with open(TRIPLETS_PATH, 'r') as f:
        triplets = json.load(f)

    # We'll collect scenes by their scene_id
    scenes = {}

    for triplet in triplets:
        for key in ['anchor', 'positive', 'negative']:
            scene = triplet[key]
            scene_id = scene['scene_id']
            if scene_id not in scenes:
                scenes[scene_id] = scene

    print(f"Found {len(scenes)} unique scenes.")

    for scene_id, scene in tqdm(scenes.items(), desc="Processing scenes"):
        scene_dir = os.path.join(OUTPUT_BASE_DIR, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        for entry in scene['entries']:
            frame_name = entry['video_id']  # e.g., nVbIUDjzWY4_102400ms
            src_path = find_frame_file(frame_name)
            if src_path is None:
                print(f"WARNING: Frame not found for {frame_name}")
                continue
            dst_path = os.path.join(scene_dir, os.path.basename(src_path))

            # Copy if not already copied
            if not os.path.isfile(dst_path):
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    prepare_scene_folders()
