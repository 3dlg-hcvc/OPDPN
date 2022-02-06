import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PC_lib.utils.box_util import get_corners_from_bbx

class PCDataset(Dataset):
    def __init__(self, data_path, num_points, max_K):
        self.f_data = h5py.File(data_path)
        self.instances = sorted(self.f_data)
        self.num_points = num_points
        self.max_K = max_K

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        id = self.instances[index]
        ins = self.f_data[id]

        # Get the points index used to sample points
        input_points = ins['camcs_per_point'][:]
        input_points_num = input_points.shape[0]
        perm = np.random.permutation(input_points_num)[:self.num_points]
        if self.num_points > input_points_num:
            additional = np.random.choice(input_points_num, self.num_points - input_points_num, replace=True)
            perm = np.concatenate((perm, additional))
        assert perm.shape[0] == self.num_points, f'{perm.shape[0]}, {self.num_points}, {input_points_num}'

        # Get the camcs_per_point
        camcs_per_point = torch.tensor(input_points, dtype=torch.float32)[perm]
        # Get all other items 
        gt_dict = {}
        gt_dict["category_per_point"] = torch.tensor(ins["category_per_point"][:], dtype=torch.float32)[perm]
        gt_dict["mtype_per_point"] = torch.tensor(ins["mtype_per_point"][:], dtype=torch.float32)[perm]
        gt_dict["maxis_per_point"] = torch.tensor(ins["maxis_per_point"][:], dtype=torch.float32)[perm]
        gt_dict["morigin_per_point"] = torch.tensor(ins["morigin_per_point"][:], dtype=torch.float32)[perm]
        
        # for each instance, the number_instance is different, but we will construct the gt corner to be K * 8 * 3
        gt_dict["num_instances"] = ins["num_instances"][0]
        gt_dict["instance_per_point"] = torch.tensor(ins["instance_per_point"][:], dtype=torch.float32)[perm]
        x = []
        y = []
        z = []
        instance_valid = []
        index = 0
        for k in range(gt_dict["num_instances"]):
            instance_index = (gt_dict["instance_per_point"] == k).nonzero(as_tuple=False)
            if instance_index.shape[0] == 0:
                instance_valid.append(False)
            else:
                instance_valid.append(True)
                x.append([torch.min(camcs_per_point[instance_index, 0]), torch.max(camcs_per_point[instance_index, 0])])
                y.append([torch.min(camcs_per_point[instance_index, 1]), torch.max(camcs_per_point[instance_index, 1])])
                z.append([torch.min(camcs_per_point[instance_index, 2]), torch.max(camcs_per_point[instance_index, 2])])
                if k != index:
                    # Remap the instance id if they are not the same
                    gt_dict["instance_per_point"][instance_index] = index
                index += 1
        x = torch.tensor(x)
        y = torch.tensor(y)
        z = torch.tensor(z)

        gt_dict["num_instances"] = np.array(instance_valid).sum()
        # Calculate the bbx size and center for K * 3
        bbx_size = torch.zeros((gt_dict["num_instances"], 3))
        bbx_center = torch.zeros((gt_dict["num_instances"], 3))
        if gt_dict["num_instances"] != 0:
            bbx_size[:, 0] = x[:, 1] - x[:, 0]
            bbx_size[:, 1] = y[:, 1] - y[:, 0]
            bbx_size[:, 2] = z[:, 1] - z[:, 0]
            bbx_center[:, 0] = (x[:, 1] + x[:, 0]) / 2
            bbx_center[:, 1] = (y[:, 1] + y[:, 0]) / 2
            bbx_center[:, 2] = (z[:, 1] + z[:, 0]) / 2
            corners = get_corners_from_bbx(bbx_size, bbx_center)
        else:
            corners = torch.zeros((0, 8, 3))

        assert self.max_K >= gt_dict["num_instances"]

        gt_dict["corners"] = torch.zeros((self.max_K, 8, 3))
        gt_dict["corners"][:gt_dict["num_instances"]] = corners

        return camcs_per_point, gt_dict, id
