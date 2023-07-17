# https://gist.github.com/HTLife/e8f1c4ff1737710e34258ef965b48344
import numpy as np
import struct
import random
from open3d import *


#===== Load Data =======================================================================
def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    list_intensity = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            list_intensity.append(intensity)
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd, list_intensity

bin_file = './data/data_object_velodyne/testing/velodyne/000100.bin' # source file path
pcd_original, intensity_original = convert_kitti_bin_to_pcd(bin_file)  # convert using function above
pcd_ndarray = np.asarray(pcd_original.points)
intensity_ndarray = np.array([np.asarray(intensity_original)]).reshape(-1,1)
pcd_ndarray = np.concatenate((pcd_ndarray, intensity_ndarray), axis=1)

# open3d.visualization.draw_geometries([pcd_original]) # show
# open3d.io.write_point_cloud('1644609772_916356000.pcd', pcd_, write_ascii=False, compressed=False, print_progress=False) # save

#===== Voxel Partition =======================================================================
def voxel_partition_and_grouping_and_random_sampling(input, det_type):
    # - input    : data (type: np array)
    # - det_type : detection type (car, ncar)
    # - output   : voxel partitioned data (type: np array)

    # ===== RANGE ====================
    if det_type == "car": # range : [-3,1] x [-40,40] x [0,70.4] (z, y, x)
        x_min, x_max = 0, 70.4
        y_min, y_max = -40, 40
        z_min, z_max = -3, 1
        T = 35
    elif det_type == "ncar":
        # range : [-3,1] x [-20,20] x [0,48] (z, y, x)
        x_min, x_max = 0, 48
        y_min, y_max = -20, 20
        z_min, z_max = -3, 1
        T = 45
        
    else:
        print("Please enter det_type correctly : \"car\" or \"ncar\"")
    check_x = (x_min <= input[:, 0]) & (x_max >= input[:, 0])
    check_y = (y_min <= input[:, 1]) & (y_max >= input[:, 1])
    check_z = (z_min <= input[:, 2]) & (z_max >= input[:, 2])
    check_arr = check_x & check_y & check_z
    check_idx = np.where(check_arr == True)[0]
    pcd_ranged = input[check_idx]

    # ===== VOXEL PARTITION ==========
    v_D, v_H, v_W = 0.4, 0.2, 0.2  # voxel size
    
    # starting coordinate of each voxel
    partition_x = [x / 10 for x in range(int(x_min * 10), int(x_max * 10), int(v_W * 10))]
    partition_y = [y / 10 for y in range(int(y_min * 10), int(y_max * 10), int(v_H * 10))]
    partition_z = [z / 10 for z in range(int(z_min * 10), int(z_max * 10), int(v_D * 10))]

    # voxel num check
    Dn, Hn, Wn = int((z_max - z_min) // v_D), int((y_max - y_min) // v_H), int((x_max - x_min) // v_W) # num of voxels along each axis
    if not (len(partition_x) != Wn & len(partition_y) != Hn & len(partition_z) != Dn):
        print("Voxel Number Error!")

    # ===== GROUPING =================
    pcd_waiting = pcd_ranged

    voxels_dict = {}
    for zz in range(len(partition_z)):
        for yy in range(len(partition_y)):
            for xx in range(len(partition_x)):
                key_name = str(zz) + "_" + str(yy) + "_" + str(xx)
                voxels_dict[key_name]=[]

    for point in pcd_waiting:
        voxidx_x = np.where((point[0] >= partition_x) == True)[0][-1]
        voxidx_y = np.where((point[1] >= partition_y) == True)[0][-1]
        voxidx_z = np.where((point[2] >= partition_z) == True)[0][-1]
        key_found = str(voxidx_z) + "_" + str(voxidx_y) + "_" + str(voxidx_x)
        voxels_dict[key_found].append(point)
        pcd_waiting = np.setdiff1d(pcd_waiting, np.array(point))
    
    # ===== RANDOM SAMPLING ==========
    for voxel_key in voxels_dict:
        if len(voxels_dict[voxel_key]) > T:
            # before = voxels_dict[voxel_key]
            # print("before voxels:", len(before))
            voxels_dict[voxel_key] = random.sample(voxels_dict[voxel_key], T)
            # after = voxels_dict[voxel_key]
            # print("after voxels:", len(after))

    return voxels_dict

#===== Stacked Voxel Feature Encoding ========================================================
def stackedVFE(input_voxels):
    # centroids = []
    V_in_all = {}

    for dictkey in input_voxels:
        voxel_current = np.array(input_voxels[dictkey])
        if len(voxel_current) == 0:
            pass
        else:
            centroid = (np.sum(voxel_current, axis=0) / len(voxel_current))[:3]
            # centroids.append(centroid)
            point_minus_centroid = voxel_current[:,:3] - centroid
            V_in_all[dictkey] = np.concatenate((voxel_current, point_minus_centroid), axis=1)
    print("here")


voxel_dictionary = voxel_partition_and_grouping_and_random_sampling(pcd_ndarray, "car")
voxel_feature = stackedVFE(voxel_dictionary)