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
from renderer import Renderer

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

rgb = np.array([[255, 255, 255], [0, 0, 0]] + ncolors(7))

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
    camcs_per_point = result["camcs_per_point"][:, :3]
    category_per_point = result[f"category_per_point"][:]

    # import pdb
    # pdb.set_trace()

    # r = Renderer(camcs_per_point, mask=category_per_point.astype(int))
    # r.show()

    # import pdb
    # pdb.set_trace()
    instance_per_point = result[f"instance_per_point"][:]
    mtype_per_point = result[f"mtype_per_point"][:]
    
    part_index = np.where(category_per_point != 1)
    base_index = np.where(category_per_point == 1)

    # import pdb
    # pdb.set_trace()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(camcs_per_point))
    pcd.colors = o3d.utility.Vector3dVector(rgb[instance_per_point.astype(int) + 2] / 255.0)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, camera])

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