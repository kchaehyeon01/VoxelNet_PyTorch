import numpy as np
import os
import open3d as o3d

# path = "./data/data_object_velodyne/testing/velodyne"

# filenames = sorted(os.listdir(path))

# points = []
# for filename in filenames:
#     point = np.fromfile(os.path.join(path,filename), dtype=np.float32).reshape(-1, 4)
#     points.append(point)

# print(len(points))



# print("Load a ply point cloud, print it, and render it")
# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])


path = "./data/data_object_velodyne/testing/velodyne/000001.bin"
pcd = o3d.io.read_point_cloud(path)
print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd])