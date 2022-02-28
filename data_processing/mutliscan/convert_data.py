import json
import h5py
import glob
import math
import numpy as np
from time import time
from PIL import Image
from alive_progress import alive_bar
import os

RAW_MODEL_PATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/dataset"
TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/PC_dataset/testIds.json'
VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/PC_dataset/validIds.json'

OUTPUTPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/multiscan_data/PC_dataset"

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

existDir(OUTPUTPATH)

CATEGORY_MAP = {"drawer": 0, "door": 1, "lid": 2, "others": 3, "seat": 4}
TYPE_MAP = {"rotation": 0, "translation": 1}
CATEGORY_NUM = 5
TYPE_NUM = 2

def addModel(model_path, h5_file, max_K=5):
    data = np.load(model_path, allow_pickle=True)['instances']
    model_name = model_path.split('/')[-1].rsplit('.', 1)[0]
    index = 0
    for model in data:
        camcs_per_point = model['pcd']
        # Process to get the instance index
        segm_per_point = model['segm']
        # The number instances is the number of moving parts
        num_instances = len(model['axes'])
        # Ranomize the order of the instnace index
        rand_perm = np.random.permutation(num_instances)
        part_map = {}
        for i in range(num_instances):
            part_map[i+1] = rand_perm[i]
        
        # Read the raw data
        part_names = model["part_names"][1:]
        mtypes = model["articulation_types"][1:]
        maxis = model["axes"]
        morigin = model["origins"]

        # Construct the things for all moving parts
        motions = {}
        for motion_index in range(num_instances):
            motion = {}
            motion["category"] = CATEGORY_MAP[part_names[motion_index]]
            motion["mtype"] = TYPE_MAP[mtypes[motion_index]]
            motion["maxis"] = maxis[motion_index]
            motion["morigin"] = morigin[motion_index]
            motion["instance"] = part_map[motion_index+1]
            motions[motion_index+1] = motion
        
        category_per_point = []
        mtype_per_point = []
        maxis_per_point = []
        morigin_per_point = []
        instance_per_point = []

        for part_index in segm_per_point:
            if part_index == 0:
                # the base part annotation
                category_per_point.append(CATEGORY_NUM) # 3 represents background
                mtype_per_point.append(-1) 
                maxis_per_point.append([0, 0, 0])
                morigin_per_point.append([0, 0, 0])
                instance_per_point.append(-1) 
                continue
            motion = motions[part_index]
            category_per_point.append(motion["category"])
            mtype_per_point.append(motion["mtype"]) 
            maxis_per_point.append(motion["maxis"])
            morigin_per_point.append(motion["morigin"])
            instance_per_point.append(motion["instance"])

        group = h5_file.create_group(f"{model_name}_{index}")
        group.create_dataset("num_instances", data=[num_instances], compression="gzip")
        group.create_dataset("camcs_per_point", data=camcs_per_point, compression="gzip")
        group.create_dataset("category_per_point", data=category_per_point, compression="gzip")
        group.create_dataset("mtype_per_point", data=mtype_per_point, compression="gzip")
        group.create_dataset("maxis_per_point", data=maxis_per_point, compression="gzip")
        group.create_dataset("morigin_per_point", data=morigin_per_point, compression="gzip")
        group.create_dataset("instance_per_point", data=instance_per_point, compression="gzip")

        index += 1

def main():
    # Load the ids in the val and test set
    # test_ids_file = open(TESTIDSPATH)
    # test_ids = json.load(test_ids_file)
    # test_ids_file.close()

    # valid_ids_file = open(VALIDIDPATH)
    # valid_ids = json.load(valid_ids_file)
    # valid_ids_file.close()
    # Load all the models from the raw data of MotionNet
    dir_paths = glob.glob(f"{RAW_MODEL_PATH}/*")

    train_output = h5py.File(f"{OUTPUTPATH}/train.h5", "w")
    # val_output = h5py.File(f"{OUTPUTPATH}/val.h5", "w")
    # test_output = h5py.File(f"{OUTPUTPATH}/test.h5", "w")

    train_output.attrs["CATEGORY_NUM"] = CATEGORY_NUM
    train_output.attrs["TYPE_NUM"] = TYPE_NUM
    
    # with alive_bar(len(dir_paths)) as bar:
    for current_dir in dir_paths:
        # model_name = current_dir.split('/')[-1]
        # if model_name in valid_ids:
        #     output = val_output
        # elif model_name in test_ids:
        #     output = test_output
        # else:
        #     output = train_output
        # addModel(current_dir, output)

        addModel(current_dir, train_output)

            # bar()

if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(f'Total time duration: {stop - start}')