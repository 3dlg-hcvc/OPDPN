import json
import h5py
import glob
import math
import numpy as np
from time import time
from PIL import Image
from alive_progress import alive_bar
import os

RAW_MODEL_PATH = {"train": "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/pointnet_baseline/baseline_train.h5", 
                "test": "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/pointnet_baseline/baseline_test.h5", 
                "val": "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/pointnet_baseline/baseline_val.h5"}

OUTPUTPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/dataset_multiscan"

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

existDir(OUTPUTPATH)

# number of moving part categories
CATEGORY_NUM = 4
TYPE_NUM = 2

def addModel(raw_data, h5_file, max_K):
    object_ids = raw_data.keys()
    ignore_index = 0
    for object_id in object_ids:
        anno = raw_data[object_id]

        # The number instances is the number of moving parts
        num_instances = anno.attrs["numParts"]
        if num_instances > max_K:
            # If it has more instances, we will just ignore
            ignore_index += 1
            print(f"[{ignore_index}] Ignore the object {object_id}")
            continue

        camcs_per_point = anno["pts"][:]
        # Make the base part category equal to the CATEGORY_NUM
        segm_per_point = anno["part_semantic_masks"][:]
        segm_per_point = segm_per_point - 1
        assert CATEGORY_NUM not in np.unique(segm_per_point)
        segm_per_point[segm_per_point == -1] = CATEGORY_NUM

        instance_per_point = anno["part_instance_masks"][:] - 1

        mtypes = anno["joint_types"][:]
        maxis = anno["joint_axes"][:]
        morigin = anno["joint_origins"][:]

        mtype_per_point = []
        maxis_per_point = []
        morigin_per_point = []
        for index in instance_per_point:
            if index == -1:
                mtype_per_point.append(-1) 
                maxis_per_point.append([0, 0, 0])
                morigin_per_point.append([0, 0, 0])
                continue
            mtype_per_point.append(mtypes[index]) 
            maxis_per_point.append(maxis[index])
            morigin_per_point.append(morigin[index])


        group = h5_file.create_group(object_id)
        group.create_dataset("num_instances", data=[num_instances], compression="gzip")
        group.create_dataset("camcs_per_point", data=camcs_per_point, compression="gzip")
        group.create_dataset("category_per_point", data=segm_per_point, compression="gzip")
        group.create_dataset("mtype_per_point", data=mtype_per_point, compression="gzip")
        group.create_dataset("maxis_per_point", data=maxis_per_point, compression="gzip")
        group.create_dataset("morigin_per_point", data=morigin_per_point, compression="gzip")
        group.create_dataset("instance_per_point", data=instance_per_point, compression="gzip")

def main():
    max_K = 10

    train_output = h5py.File(f"{OUTPUTPATH}/train.h5", "w")
    val_output = h5py.File(f"{OUTPUTPATH}/val.h5", "w")
    test_output = h5py.File(f"{OUTPUTPATH}/test.h5", "w")

    train_output.attrs["CATEGORY_NUM"] = CATEGORY_NUM
    train_output.attrs["TYPE_NUM"] = TYPE_NUM

    for traintest in RAW_MODEL_PATH.keys():
        # Load raw_data
        raw_data = h5py.File(RAW_MODEL_PATH[traintest])
        
        if traintest == "val":
            output = val_output
        elif traintest == "test":
            output = test_output
        else:
            output = train_output
        addModel(raw_data, output, max_K)


if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(f'Total time duration: {stop - start}')