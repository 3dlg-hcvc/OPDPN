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

OUTPUTPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/visualization_real"

rgb = np.array([[0, 0, 0], [255, 255, 255]] + ncolors(5))
intrinsic_matrix = np.array([[283.18526475694443, 0., 126.65098741319443], [0., 283.18526475694443, 128.45118272569442],[ 0., 0., 1.]])   


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
    category_per_point = result[f"category_per_point"][:]
    instance_per_point = result[f"instance_per_point"][:]
    mtype_per_point = result[f"mtype_per_point"][:]
    instance_img = np.zeros((256, 256, 3))
    
    point_2d = np.dot(intrinsic_matrix, camcs_per_point[:, :3].T).T
    new_x = (point_2d[:, 0] / point_2d[:, 2]).astype(int)
    new_y = (point_2d[:, 1] / point_2d[:, 2]).astype(int)

    part_index = np.where(category_per_point != 3)
    base_index = np.where(category_per_point == 3)
    # import pdb
    # pdb.set_trace()

    # instance_img[new_x[base_index], new_y[base_index]] = rgb[1]
    instance_img[new_y[part_index], new_x[part_index]] = rgb[instance_per_point[part_index].astype(int) + 2]
    image = Image.fromarray(np.uint8(instance_img))
    image = image.convert('RGB')
    image.save(f"{output_dir}/instance.png")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(camcs_per_point))
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, camera])
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True) #works for me with False, on some systems needs to be true
    # vis.add_geometry(pcd)
    # vis.update_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(f"{output_dir}/o3d.png")
    # vis.destroy_window()


if __name__ == "__main__":
    start = time()
    args = get_parser().parse_args()

    results = h5py.File(args.result_path)
    instances = results.keys()

    # with alive_bar(len(instances)) as bar:
    for instance in instances:
        print(instance)
        renderResults(instance, results[instance], "gt")
        # renderResults(instance, results[instance], "pred")



    stop = time()
    print(f'Total time duration: {stop - start}')