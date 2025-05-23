import os
import json
import numpy as np
from read_write_model import read_model

def qvec2rotmat(qvec):
    # Quaternion to rotation matrix
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def convert_scene_to_transforms(scene_path, image_folder='images'):
    sparse_path = os.path.join(scene_path, 'sparse', '0')
    if not os.path.exists(sparse_path):
        print(f"⚠️ Sparse folder missing: {sparse_path}")
        return
    
    cameras, images, _ = read_model(sparse_path, ext='.bin')

    camera = list(cameras.values())[0]
    fx = camera.params[0]
    fy = camera.params[1] if len(camera.params) > 1 else fx
    cx = camera.params[2]
    cy = camera.params[3]

    w = camera.width
    h = camera.height
    camera_angle_x = 2 * np.arctan(w / (2 * fx))

    frames = []
    for image_id, image in images.items():
        fname = os.path.basename(image.name)
        transform = np.eye(4)
        rot = qvec2rotmat(image.qvec)
        transform[:3, :3] = rot.T
        transform[:3, 3] = -rot.T @ image.tvec
        frames.append({
            "file_path": os.path.join(image_folder, fname).replace('\\', '/'),
            "transform_matrix": transform.tolist()
        })

    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }

    # Save transforms.json
    out_path = os.path.join(scene_path, 'transforms.json')
    with open(out_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"✅ Saved: {out_path}")

def process_all_scenes(base_dir=r'C:\Aneesh\EDI VI\data\final_colmap_outputs'):
    for scene_name in os.listdir(base_dir):
        scene_path = os.path.join(base_dir, scene_name)
        convert_scene_to_transforms(scene_path)

if __name__ == '__main__':
    process_all_scenes()
