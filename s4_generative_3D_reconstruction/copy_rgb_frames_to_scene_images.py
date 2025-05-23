import os
import json
import shutil

# Paths
SCENE_FRAMES_DIR = r"C:\Aneesh\EDI VI\data\scene_frames"  # Each subfolder here is scene_id/
FINAL_COLMAP_DIR = r"C:\Aneesh\EDI VI\data\final_colmap_outputs"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def copy_valid_frames(scene_dir):
    transforms_path = os.path.join(scene_dir, "transforms.json")
    scene_id = os.path.basename(scene_dir)
    scene_frames_path = os.path.join(SCENE_FRAMES_DIR, scene_id)

    if not os.path.exists(transforms_path):
        print(f"⚠️ No transforms.json found for {scene_id}, skipping.")
        return

    if not os.path.exists(scene_frames_path):
        print(f"❌ No RGB frame folder found for {scene_id}, skipping.")
        return

    with open(transforms_path, 'r') as f:
        data = json.load(f)

    images_out_dir = os.path.join(scene_dir, "images")
    ensure_dir(images_out_dir)

    copied = 0
    for frame in data['frames']:
        fname = os.path.basename(frame['file_path'])
        src_path = os.path.join(scene_frames_path, fname)
        dst_path = os.path.join(images_out_dir, fname)

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            copied += 1
        else:
            print(f"❗ Missing expected frame: {src_path}")

    print(f"✅ {scene_id}: Copied {copied} frames to {images_out_dir}")

def run():
    for scene_id in os.listdir(FINAL_COLMAP_DIR):
        scene_path = os.path.join(FINAL_COLMAP_DIR, scene_id)
        if os.path.isdir(scene_path):
            copy_valid_frames(scene_path)

if __name__ == "__main__":
    run()
