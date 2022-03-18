import h5py
import argparse
import numpy as np
from time import time
from PIL import Image
from alive_progress import alive_bar
import os
import math
import open3d as o3d



import colorsys
import random

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num 
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors 
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

OUTPUTPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/visualization"

rgb = np.array([[255, 255, 255], [255, 255, 255], [248, 11, 11], [248, 163,  35], [ 64, 251,  17], [204,  29, 247]] + ncolors(10))


def getFocalLength(FOV, height, width=None):
    # Used to calculate the fixed intrinsic parameters for motionnet
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)

FOV = 50
img_width = 256
img_height = 256
fx, fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
cy = img_height / 2
cx = img_width / 2


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--result_path",
        default=None,
        metavar="FILE",
        help="hdf5 file which contains the test results",
    )
    return parser

def renderResults(instance, result, prefix):
    existDir(f"{OUTPUTPATH}/{instance}")
    existDir(f"{OUTPUTPATH}/{instance}/{prefix}")
    output_dir = f"{OUTPUTPATH}/{instance}/{prefix}"
    camcs_per_point = result["camcs_per_point"][:]
    category_per_point = result[f"{prefix}_category_per_point"][:]
    instance_per_point = result[f"{prefix}_instance_per_point"][:]
    mtype_per_point = result[f"{prefix}_mtype_per_point"][:]
    instance_img = np.ones((img_width, img_height, 3)) * 255
    x = camcs_per_point[:, 0]
    y = camcs_per_point[:, 1]
    z = camcs_per_point[:, 2]
    # import pdb
    # pdb.set_trace()
    new_x = (x * fx / (-z) + cx).astype(int)
    new_y = (-(y * fy / (-z)) + cy).astype(int)
    # part_index = np.where(category_per_point != 3)
    # base_index = np.where(category_per_point == 3)
    # x_min = np.min(new_x[part_index])
    # x_max = np.max(new_x[part_index])
    # y_min = np.min(new_y[part_index])
    # y_max = np.max(new_y[part_index])
    # instance_img[x_min, y_min] = rgb[2]
    # instance_img[x_max, y_max] = rgb[3]
    # import pdb
    # pdb.set_trace()

    instance_img[new_x, new_y] = rgb[1]
    # instance_img[new_y[part_index], new_x[part_index]] = rgb[instance_per_point[part_index].astype(int) + 2]
    image = Image.fromarray(np.uint8(instance_img))
    image = image.convert('RGB')
    image.save(f"{output_dir}/instance.png")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(camcs_per_point))
    # import pdb
    # pdb.set_trace()
    print(rgb)
    pcd.colors = o3d.utility.Vector3dVector(rgb[category_per_point.astype(int) + 2] / 255)
    # import pdb
    # pdb.set_trace()
    # pcd.colors = o3d.utility.Vector3dVector(rgb[np.ones(category_per_point.shape[0]).astype(int)] / 255)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    start = time()
    args = get_parser().parse_args()

    results = h5py.File(args.result_path)
    instances = results.keys()

    index = 0
    # with alive_bar(len(instances)) as bar:
    for instance in instances:
        if instance != "46981-0-2":
            continue
        # if index == 10:
        #     break
        # renderResults(instance, results[instance], "gt")
        renderResults(instance, results[instance], "gt")
        index += 1



    stop = time()
    print(f'Total time duration: {stop - start}')