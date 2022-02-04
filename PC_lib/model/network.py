import torch
import torch.nn as nn
import torch.nn.functional as F
from PC_lib.model.backbone import PointNet2
from PC_lib.model import loss

class PC_BASELINE(nn.Module):
    def __init__(self, max_K, category_number):
        super().__init__()
        self.max_K = max_K
        self.category_number = category_number
        # Define the shared PN++
        self.backbone = PointNet2()
        # The binary classification layer for judging if a point belongs to the moving parts
        # 0: not belongs to the moving part, 1: belongs to the moving part
        self.cls_layer = nn.Conv1d(128, 2, kernel_size=1, padding=0)
        # Predict the semantic segmentation (part category) for each point
        self.category_layer = nn.Conv1d(128, category_number, kernel_size=1, padding=0)
        # The instance labelling layer for judging which instance a point belongs to
        # 0 ~ k-1: the instance index -> it should be unordered (when calculating the loss, need to match the gt with the pred instance)
        self.instance_layer = nn.Conv1d(128, self.max_K, kernel_size=1, padding=0)
        # The layers for motion prediction: motion type, motion axis and motion origin
        # For motion type: it's binary -> 0: rotation, 1: translation
        # For motion axis: a vector of length three in the camera coordinate
        # For motion origin, a vector of length three in the camera coordinate
        self.motion_feature_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.mtype_layer = nn.Conv1d(128, 2, kernel_size=1, padding=0)
        self.maxis_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
        self.morigin_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)

    def forward(self, input):
        features = self.backbone(input)
        pred_cls_per_point = self.cls_layer(features)
        pred_category_per_point = self.category_layer(features)
        pred_instance_per_point = self.instance_layer(features)

        motion_features = self.motion_feature_layer(features)
        pred_mtype_per_point = self.mtype_layer(motion_features)
        pred_maxis_per_point = self.maxis_layer(motion_features)
        pred_morigin_per_point = self.morigin_layer(motion_features)

        pred = {
            "cls_per_point": pred_cls_per_point,
            "category_per_point": pred_category_per_point,
            "instance_per_point": pred_instance_per_point,
            "mtype_per_point": pred_mtype_per_point,
            "maxis_per_point": pred_maxis_per_point,
            "morigin_per_point": pred_morigin_per_point,
        }

        return pred

    def losses(self, pred, gt):
        # Convert the gt["cls_per_point"] into B*N*2
        gt_cls_per_point = F.one_hot(gt["cls_per_point"].long(), num_classes=2)
        cls_loss = loss.compute_miou_loss(pred["cls_per_point"], gt_cls_per_point)
        # Convert the gt["category_per_point"] into B*N*2
        gt_category_per_point = F.one_hot(gt["category_per_point"].long(), num_classes=self.category_number)
        category_loss = loss.compute_miou_loss(pred["category_per_point"], gt_category_per_point)
        # Convert the gt["instance_per_point"] into B*N*max_K -> This one need further matching
        # gt_instance_per_point = F.one_hot(gt["instance_per_point"].long(), num_classes=self.max_K)
        # instance_loss = loss.compute_miou_loss(pred["instance_per_point"], gt_instance_per_point)
        # # 

