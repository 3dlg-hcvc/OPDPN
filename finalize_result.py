import h5py
import argparse
import numpy as np
import math
from alive_progress import alive_bar

def getFocalLength(FOV, height, width=None):
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

def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--result_path",
        default=None,
        metavar="FILE",
        help="hdf5 file which contains the test results",
    )
    parser.add_argument(
        "--output",
        default="/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/pc_output",
        metavar="Dir",
        help="output dir used to store the output final_result.h5",
    )
    parser.add_argument(
        "--max_K",
        default=5,
        type=int,
        help="indicatet the max number for the segmentation",
    )
    return parser

def convert_result(instance, group, max_K):
    global fx, fy, cx, cy
    camcs_per_point = np.array(instance["camcs_per_point"][:])
    pred_category_per_point = np.array(instance["pred_category_per_point"][:])
    pred_instance_per_point = np.array(instance["pred_instance_per_point"][:])
    pred_maxis_per_point = np.array(instance["pred_maxis_per_point"][:])
    pred_morigin_per_point = np.array(instance["pred_morigin_per_point"][:])
    pred_mtype_per_point = np.array(instance["pred_mtype_per_point"][:])

    num_moving_point = np.where(pred_category_per_point != 3)[0].shape[0]
    # 3 is the index for the base part
    is_valid = []
    cat_map = []
    mtype_map = []
    maxis_map = []
    morigin_map = []
    bbx_map = []
    
    for index in range(max_K):
        # Judge if the instance is valid
        instance_index = np.where((pred_category_per_point != 3) * (pred_instance_per_point == index))[0]
        if instance_index.shape[0] <= 0.1 * num_moving_point:
            is_valid.append(False)
            cat_map.append(-1)
            mtype_map.append(-1)
            maxis_map.append([-1, -1, -1])
            morigin_map.append([-1, -1, -1])
            bbx_map.append([-1, -1, -1 , -1])
            continue
        is_valid.append(True)
        cat_map.append(np.bincount(pred_category_per_point[instance_index]).argmax())
        mtype_map.append(np.bincount(pred_mtype_per_point[instance_index]).argmax())
        maxis_map.append(np.median(pred_maxis_per_point[instance_index], 0))
        morigin_map.append(np.median(pred_morigin_per_point[instance_index], 0))

        bbx_cam = camcs_per_point[instance_index]
        bbx_cam[:, 0] = bbx_cam[:, 0] * fx / (-bbx_cam[:, 2]) + cx
        bbx_cam[:, 1] = -(bbx_cam[:, 1] * fy / (-bbx_cam[:, 2])) + cy
        bbx_cam[:, 2] = bbx_cam[:, 2]
        x_min = np.float64(np.min(bbx_cam[:, 0]))
        x_max = np.float64(np.max(bbx_cam[:, 0]))
        y_min = np.float64(np.min(bbx_cam[:, 1]))
        y_max = np.float64(np.max(bbx_cam[:, 1]))
        bbx_map.append([x_min, y_min, x_max - x_min, y_max - y_min])

    group.create_dataset(
        "camcs_per_point",
        data=np.array(camcs_per_point),
        compression="gzip",
    )
    group.create_dataset(
        "is_valid",
        data=np.array(is_valid),
        compression="gzip",
    )
    group.create_dataset(
        "cat_map",
        data=np.array(cat_map),
        compression="gzip",
    )
    group.create_dataset(
        "mtype_map",
        data=np.array(mtype_map),
        compression="gzip",
    )
    group.create_dataset(
        "maxis_map",
        data=np.array(maxis_map),
        compression="gzip",
    )
    group.create_dataset(
        "morigin_map",
        data=np.array(morigin_map),
        compression="gzip",
    )
    group.create_dataset(
        "bbx_map",
        data=np.array(bbx_map),
        compression="gzip",
    )

if __name__ == "__main__":
    args = get_parser().parse_args()

    results = h5py.File(args.result_path)
    instances = results.keys()

    final_results = h5py.File(f"{args.output}/final_result.h5", "w")
    with alive_bar(len(instances)) as bar:
        for instance in instances:
            group = final_results.create_group(instance)
            convert_result(results[instance], group, args.max_K)
            bar()
        