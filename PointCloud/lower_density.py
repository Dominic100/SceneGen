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
    
    # Sample N random points on the surface - reduce number for compatibility
    N = 20000  # Lower point count for better performance
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
    
    # Create simplified PLY file directly using manual writing
    # This ensures maximum compatibility with Three.js
    sparse_ply_path = "sparse_point_cloud.ply"
    write_compatible_ply(sparse_ply_path, points, colors)
    
    # Copy PLY file to the target location
    copy_to_target_location(sparse_ply_path)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def write_compatible_ply(filename, points, colors):
    """Write a simplified PLY file that's guaranteed to be compatible with Three.js"""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"Written compatible PLY file to {filename}")

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
    print(f"File exists after copy: {os.path.exists(target_path)}")
    print(f"File size: {os.path.getsize(target_path)} bytes")
    print(f"This will be used by the server as the 3D model result")
    
    # Verify the target file is readable as text
    try:
        with open(target_path, 'r') as f:
            first_few_lines = [f.readline() for _ in range(5)]
            print("First few lines of target file:")
            print(''.join(first_few_lines))
    except UnicodeDecodeError:
        print("WARNING: Target file is not in ASCII format - may cause rendering issues")

if __name__ == "__main__":
    main()