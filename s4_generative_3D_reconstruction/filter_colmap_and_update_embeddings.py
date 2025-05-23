import os
import shutil
import pandas as pd

# Paths
colmap_output_dir = r'C:\Aneesh\EDI VI\data\colmap_outputs'
final_output_dir = r'C:\Aneesh\EDI VI\data\final_colmap_outputs'
scene_embeddings_csv = r'C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\scene_embeddings\scene_embeddings.csv'
filtered_embeddings_csv = r'C:\Aneesh\EDI VI\s3_cross_modal_fusion_improved\scene_embeddings\scene_embeddings_filtered.csv'

# Create final output directory
os.makedirs(final_output_dir, exist_ok=True)

# List of scene IDs with valid COLMAP outputs
valid_scenes = []

# Check each scene in colmap_output
for scene_name in os.listdir(colmap_output_dir):
    scene_path = os.path.join(colmap_output_dir, scene_name)
    sparse_path = os.path.join(scene_path, 'sparse')

    # Check if sparse/ exists and is non-empty
    if os.path.isdir(sparse_path) and len(os.listdir(sparse_path)) > 0:
        # Copy entire scene to final_colmap_output
        dest_path = os.path.join(final_output_dir, scene_name)
        shutil.copytree(scene_path, dest_path, dirs_exist_ok=True)

        # Record the scene ID
        valid_scenes.append(scene_name)

print(f"✔️ {len(valid_scenes)} valid scenes copied to '{final_output_dir}'.")

# Load scene_embeddings.csv and filter rows
df = pd.read_csv(scene_embeddings_csv)

# Keep only rows where scene_id is in valid_scenes
df_filtered = df[df['scene_id'].isin(valid_scenes)]

# Save filtered version
df_filtered.to_csv(filtered_embeddings_csv, index=False)
print(f"📝 Filtered embeddings saved to '{filtered_embeddings_csv}' with {len(df_filtered)} scenes.")