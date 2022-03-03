import json
import h5py
import glob
import math
import numpy as np
from time import time
from PIL import Image
from alive_progress import alive_bar
import os

RAW_MODEL_PATH = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/raw_data_real"
TESTIDSPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/testIds_real.json'
VALIDIDPATH = '/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/raw_data_process/preprocess/validIds_real.json'
OBJECT_MASK_PATH = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/real_object_mask"

name_map_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/real_scan_process/real_name.json"
with open(name_map_path) as f:
    name_map = json.load(f)
reverse_name_map = {value : key for (key, value) in name_map.items()}

OUTPUTPATH = "/localhome/hja40/Desktop/Research/proj-motionnet/PC_motion_prediction/dataset_real"

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

existDir(OUTPUTPATH)

CATEGORY_MAP = {"drawer": 0, "door": 1, "lid": 2}
TYPE_MAP = {"rotation": 0, "translation": 1}
CATEGORY_NUM = 3
TYPE_NUM = 2

intrinsic_matrix = np.array([[283.18526475694443, 0., 126.65098741319443], [0., 283.18526475694443, 128.45118272569442],[ 0., 0., 1.]])    

def saveMask(img):
    # This function is only used for debugging
    image = Image.fromarray((img+2)*30)
    image = image.convert('RGB')
    image.save("./test.png")

def addModel(model_path, h5_file, max_K=5):
    rgb_paths = glob.glob(f"{model_path}/origin/*.png")
    for rgb_path in rgb_paths:
        rgb_name = rgb_path.split('/')[-1].split('.')[0]
        # Read the corresponding depth image
        depth = np.array(Image.open(f"{model_path}/depth/{rgb_name}_d.png")) * 1.0 / 1000
        img_size = depth.shape
        # Read the masks: -2: background, -1: base_part, 0-K: moving part
        mask = np.zeros(img_size) - 2
        mask_paths = glob.glob(f"{model_path}/mask/{rgb_name}_*.png")
        if len(mask_paths) > max_K:
            continue
        # Get the mask of the whole object in the image
        object_id = rgb_path.split('/')[-3]
        image_name = f"object_{reverse_name_map[rgb_name.split('-')[0]]}-{rgb_name.split('-')[1]}"
        object_mask = np.array(Image.open(f"{OBJECT_MASK_PATH}/{object_id}/{image_name}.png")).sum(2)
        mask[np.where(object_mask != 765)] = -1

        part_indexes = []
        for mask_path in mask_paths:
            mask_index = mask_path.split('/')[-1].split('.')[0].split('_')[-1]
            part_indexes.append(mask_index)
            raw_mask = np.array(Image.open(mask_path))
            mask[np.where(raw_mask == True)] = mask_index
    
        # Map the part indexes to random instance id to make the part instance fully unordered
        # This will be the instance id for each part, starting from 0
        if (depth[np.where(mask > -2)] == 0).sum() != 0:
            continue
        num_instances = len(part_indexes)
        rand_perm = np.random.permutation(num_instances)
        part_map = {}
        for i in range(num_instances):
            part_map[part_indexes[i]] = rand_perm[i]
        
        # Record the motion information
        motions = {}
        with open(f"{model_path}/origin_annotation/{rgb_name}.json") as f:
            annotation = json.load(f)
        for anno in annotation["motions"]:
            motion = {}
            partId = anno["partId"]
            motion["category"] = CATEGORY_MAP[anno["label"].strip()]
            motion["mtype"] = TYPE_MAP[anno["type"].strip()]
            motion["maxis"] = anno["current_axis"]
            motion["morigin"] = anno["current_origin"]
            motion["instance"] = part_map[partId]
            motions[int(partId)] = motion
        
        # Start to convert the depth into point cloud
        # Pick the points that are not the background
        y, x = np.where(mask > -2)
        z = depth[y, x]
        old_point =  np.column_stack((x.astype(float)*z, y.astype(float)*z, z))
        camcs_per_point = np.dot(np.linalg.inv(intrinsic_matrix), old_point.T).T
        point_mask = mask[y, x]
        # Begin to prepare the final data
        category_per_point = []
        mtype_per_point = []
        maxis_per_point = []
        morigin_per_point = []
        instance_per_point = []
        for index in point_mask:
            if index == -1:
                # the base part annotation
                category_per_point.append(CATEGORY_NUM) # 3 represents background
                mtype_per_point.append(-1) 
                maxis_per_point.append([0, 0, 0])
                morigin_per_point.append([0, 0, 0])
                instance_per_point.append(-1) 
                continue
            motion = motions[index]
            category_per_point.append(motion["category"])
            mtype_per_point.append(motion["mtype"]) 
            maxis_per_point.append(motion["maxis"])
            morigin_per_point.append(motion["morigin"])
            instance_per_point.append(motion["instance"])
        
        group = h5_file.create_group(rgb_name)
        group.create_dataset("num_instances", data=[num_instances], compression="gzip")
        group.create_dataset("camcs_per_point", data=camcs_per_point, compression="gzip")
        group.create_dataset("category_per_point", data=category_per_point, compression="gzip")
        group.create_dataset("mtype_per_point", data=mtype_per_point, compression="gzip")
        group.create_dataset("maxis_per_point", data=maxis_per_point, compression="gzip")
        group.create_dataset("morigin_per_point", data=morigin_per_point, compression="gzip")
        group.create_dataset("instance_per_point", data=instance_per_point, compression="gzip")

def main():
    # Load the ids in the val and test set
    test_ids_file = open(TESTIDSPATH)
    test_ids = json.load(test_ids_file)
    test_ids_file.close()

    valid_ids_file = open(VALIDIDPATH)
    valid_ids = json.load(valid_ids_file)
    valid_ids_file.close()
    # Load all the models from the raw data of MotionNet
    dir_paths = glob.glob(f"{RAW_MODEL_PATH}/*")

    train_output = h5py.File(f"{OUTPUTPATH}/train.h5", "w")
    val_output = h5py.File(f"{OUTPUTPATH}/val.h5", "w")
    test_output = h5py.File(f"{OUTPUTPATH}/test.h5", "w")

    train_output.attrs["CATEGORY_NUM"] = CATEGORY_NUM
    train_output.attrs["TYPE_NUM"] = TYPE_NUM
    # index = 0
    with alive_bar(len(dir_paths)) as bar:
        for current_dir in dir_paths:
            # index += 1
            # if index == 3:
            #     break
            model_name = current_dir.split('/')[-1]
            if model_name in valid_ids:
                output = val_output
            elif model_name in test_ids:
                output = test_output
            else:
                output = train_output
            addModel(current_dir, output)
            bar()

if __name__ == "__main__":
    start = time()

    main()

    stop = time()
    print(f'Total time duration: {stop - start}')