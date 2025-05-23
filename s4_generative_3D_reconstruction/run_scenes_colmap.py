import os
import subprocess
from tqdm import tqdm

# Update paths
SCENES_BASE_DIR = r"C:\Aneesh\EDI VI\data\scene_frames"
COLMAP_OUTPUT_BASE = r"C:\Aneesh\EDI VI\data\colmap_outputs"

COLMAP_EXE = "colmap"  # Make sure colmap is in your PATH, or provide full path here

def run_colmap_on_scene(scene_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.join(output_folder, "database.db")

    # 1. Feature extraction
    cmd_extract = [
        COLMAP_EXE, "feature_extractor",
        f"--database_path={db_path}",
        f"--image_path={scene_folder}"
    ]

    # 2. Feature matching
    cmd_match = [
        COLMAP_EXE, "exhaustive_matcher",
        f"--database_path={db_path}"
    ]

    # 3. Sparse reconstruction (mapper)
    sparse_folder = os.path.join(output_folder, "sparse")
    os.makedirs(sparse_folder, exist_ok=True)

    cmd_mapper = [
        COLMAP_EXE, "mapper",
        f"--database_path={db_path}",
        f"--image_path={scene_folder}",
        f"--output_path={sparse_folder}"
    ]

    # Run commands sequentially, print progress
    print(f"Processing scene: {os.path.basename(scene_folder)}")

    subprocess.run(cmd_extract, check=True)
    subprocess.run(cmd_match, check=True)
    subprocess.run(cmd_mapper, check=True)

def main():
    scenes = [d for d in os.listdir(SCENES_BASE_DIR) if os.path.isdir(os.path.join(SCENES_BASE_DIR, d))]

    for scene in tqdm(scenes, desc="Running COLMAP on scenes"):
        print(f"Processing scene: {scene}")
        scene_path = os.path.join(SCENES_BASE_DIR, scene)
        output_path = os.path.join(COLMAP_OUTPUT_BASE, scene)
        try:
            run_colmap_on_scene(scene_path, output_path)
        except subprocess.CalledProcessError as e:
            print(f"Error processing scene {scene}: {e}")

if __name__ == "__main__":
    main()
