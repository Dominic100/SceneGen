import trimesh
import numpy as np
import open3d as o3d
import os
import hashlib
import shutil

def main():
    # Load the OBJ file
    mesh_or_scene = trimesh.load('classroom.obj')
    
    # Check if it's a scene (multiple meshes)
    if isinstance(mesh_or_scene, trimesh.Scene):
        # Combine all meshes into a single mesh
        mesh = trimesh.util.concatenate([geom for geom in mesh_or_scene.geometry.values()])
    else:
        mesh = mesh_or_scene
    
    # Sample N random points on the surface
    N = 100000  # adjust for sparsity
    points, face_indices = trimesh.sample.sample_surface(mesh, N)
    
    # Calculate vertex colors based on position (for visualization)
    # Normalize points to 0-1 range for color mapping
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    norm_points = (points - min_bound) / (max_bound - min_bound)
    
    # Map normalized coordinates to colors
    colors = np.zeros((N, 3), dtype=np.uint8)
    colors[:, 0] = norm_points[:, 0] * 255  # R
    colors[:, 1] = norm_points[:, 1] * 255  # G
    colors[:, 2] = norm_points[:, 2] * 255  # B
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    
    # Save as PLY
    sparse_ply_path = "sparse_point_cloud.ply"
    o3d.io.write_point_cloud(sparse_ply_path, pcd)
    
    # Copy PLY file to the target location
    copy_to_target_location(sparse_ply_path)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def copy_to_target_location(ply_path):
    """Copy PLY file to the appropriate location for the server"""
    # Calculate the hash for the filename
    sys_identifier = "video-to-3d-hidden-asset"
    sys_hash = hashlib.md5(sys_identifier.encode()).hexdigest()[:8]
    target_filename = f"{sys_hash}.ply"
    
    # Determine server directory path
    server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SceneGen", "server"))
    assets_dir = os.path.join(server_dir, "assets")
    
    # Create assets directory if it doesn't exist
    os.makedirs(assets_dir, exist_ok=True)
    
    # Target path
    target_path = os.path.join(assets_dir, target_filename)
    
    # Copy the file
    shutil.copy2(ply_path, target_path)
    print(f"Point cloud copied to: {target_path}")
    print(f"This will be used by the server as the 3D model result")

if __name__ == "__main__":
    main()