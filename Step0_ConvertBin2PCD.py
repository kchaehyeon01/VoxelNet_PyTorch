# https://gist.github.com/HTLife/e8f1c4ff1737710e34258ef965b48344
import numpy as np
import struct
from open3d import *

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd

# source
bin_file = './data/data_object_velodyne/testing/velodyne/000010.bin'

# convert
pcd_ = convert_kitti_bin_to_pcd(bin_file)
pcd_nparray = np.asarray(pcd_.points)

# show
# print(pcd_)
print(pcd_nparray)
open3d.visualization.draw_geometries([pcd_])

# # save
# open3d.io.write_point_cloud('1644609772_916356000.pcd', pcd_, write_ascii=False, compressed=False, print_progress=False)