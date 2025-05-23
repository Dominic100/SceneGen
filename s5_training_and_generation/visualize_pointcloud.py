import open3d as o3d
import sys

def visualize_pointcloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_pointcloud.py <path_to_ply_file>")
        exit(1)
    visualize_pointcloud(sys.argv[1])
