import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from PC_lib.model.backbone import PointNet2
from PC_lib.model import loss
from PC_lib.utils.box_util import generalized_box3d_iou, get_corners_from_bbx

class PC_BASELINE(nn.Module):
    def __init__(self, max_K, category_number):
        super().__init__()
        self.max_K = max_K
        self.category_number = category_number
        # Define the shared PN++
        self.backbone = PointNet2()
        # Predict the semantic segmentation (part category) for each point
        # 0 - cat_num-1: different part categories; cat_num: background
        self.category_layer = nn.Conv1d(128, category_number+1, kernel_size=1, padding=0)
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
        pred_category_per_point = self.category_layer(features).transpose(1, 2)
        pred_instance_per_point = self.instance_layer(features).transpose(1, 2)

        motion_features = self.motion_feature_layer(features)
        pred_mtype_per_point = self.mtype_layer(motion_features).transpose(1, 2)
        pred_maxis_per_point = self.maxis_layer(motion_features).transpose(1, 2)
        pred_morigin_per_point = self.morigin_layer(motion_features).transpose(1, 2)

        pred_category_per_point = F.softmax(pred_category_per_point, dim=2)
        pred_instance_per_point = F.softmax(pred_instance_per_point, dim=2)

        pred_mtype_per_point = F.softmax(pred_mtype_per_point, dim=2)
        pred_maxis_per_point = F.normalize(pred_maxis_per_point, p=2, dim=2)

        pred = {
            "category_per_point": pred_category_per_point,
            "instance_per_point": pred_instance_per_point,
            "mtype_per_point": pred_mtype_per_point,
            "maxis_per_point": pred_maxis_per_point,
            "morigin_per_point": pred_morigin_per_point,
        }

        return pred

    def losses(self, pred, camcs_per_point, gt):
        # camcs_per_point is the camera coordinate used to calculate the bbx for each instance -> B * N * 3
        # Convert the gt["category_per_point"] into B*N*(category_num+1)
        gt_category_per_point = F.one_hot(gt["category_per_point"].long(), num_classes=self.category_number+1)
        loss_category = loss.compute_miou_loss(pred["category_per_point"], gt_category_per_point)
        # Matching between the predicted instances and gt_isntances, ignore the points that don't belong to the moving parts
        # Here use all predicted points to do the matching -> this will make better instance prediction
        # Calculate the GIoUs between the predicted instances and Ground truth isntances
        # pred_x, pred_y and pred_z is used to record the min and x value -> B * K * 2
        # Below code is only for calculating giou for matching
        with torch.no_grad():
            batch_size = pred["instance_per_point"].shape[0]
            pred_num_instances = self.max_K
            pred_x = torch.zeros((batch_size, pred_num_instances, 2), device=pred["instance_per_point"].device)
            pred_y = torch.zeros((batch_size, pred_num_instances, 2), device=pred["instance_per_point"].device)
            pred_z = torch.zeros((batch_size, pred_num_instances, 2), device=pred["instance_per_point"].device)
            # Calculate the mask for each instance id -> B * N
            pred_category_id_per_point = torch.argmax(pred["category_per_point"], dim=2)
            pred_instance_id_per_point = torch.argmax(pred["instance_per_point"], dim=2)
            for b in range(batch_size):
                for k in range(pred_num_instances):
                    instance_index = (torch.logical_and(pred_instance_id_per_point[b, :] == k, gt["category_per_point"][b, :] != self.category_number)).nonzero(as_tuple=False)
                    if instance_index.shape[0] == 0:
                        # If there is not points belong to this instance, just ignore it, whose box size will be 0
                        continue
                    pred_x[b, k, 0] = torch.min(camcs_per_point[b, instance_index, 0])
                    pred_x[b, k, 1] = torch.max(camcs_per_point[b, instance_index, 0])
                    pred_y[b, k, 0] = torch.min(camcs_per_point[b, instance_index, 1])
                    pred_y[b, k, 1] = torch.max(camcs_per_point[b, instance_index, 1])
                    pred_z[b, k, 0] = torch.min(camcs_per_point[b, instance_index, 2])
                    pred_z[b, k, 1] = torch.max(camcs_per_point[b, instance_index, 2])
            # Calculate the bbx size and center for B * K * 3
            pred_bbx_size = torch.zeros((batch_size, pred_num_instances, 3), device=pred["instance_per_point"].device)
            pred_bbx_center = torch.zeros((batch_size, pred_num_instances, 3), device=pred["instance_per_point"].device)
            pred_bbx_size[:, :, 0] = pred_x[:, :, 1] - pred_x[:, :, 0]
            pred_bbx_size[:, :, 1] = pred_y[:, :, 1] - pred_y[:, :, 0]
            pred_bbx_size[:, :, 2] = pred_z[:, :, 1] - pred_z[:, :, 0]
            pred_bbx_center[:, :, 0] = (pred_x[:, :, 1] + pred_x[:, :, 0]) / 2
            pred_bbx_center[:, :, 1] = (pred_y[:, :, 1] + pred_y[:, :, 0]) / 2
            pred_bbx_center[:, :, 2] = (pred_z[:, :, 1] + pred_z[:, :, 0]) / 2
            pred_corners = get_corners_from_bbx(pred_bbx_size, pred_bbx_center)
            # Calculate the GIoU among all prediction and all gts -> B * K * gt_num_instance
            gious = generalized_box3d_iou(pred_corners, gt["corners"], gt["num_instances"])
            cost = -gious.detach().cpu().numpy()
            # matching based on the giou
            assignment = []
            for b in range(batch_size):
                assign = linear_sum_assignment(cost[b, :, :gt["num_instances"][b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred["instance_per_point"].device)
                    for x in assign
                ]
                assignment.append(assign)

        # Convert the gt["mtype_per_point"] into B*N*2
        loss_instance = torch.tensor(0.0, device=pred["instance_per_point"].device)
        loss_mtype = torch.tensor(0.0, device=pred["instance_per_point"].device)
        loss_maxis = torch.tensor(0.0, device=pred["instance_per_point"].device)
        loss_morigin = torch.tensor(0.0, device=pred["instance_per_point"].device)
        for b in range(batch_size):
            moving_mask = (gt["category_per_point"][b, :] != self.category_number)

            # Below code is for the loss of instance segmentation
            gt_instance_per_point = gt["instance_per_point"][b, moving_mask].clone()
            instance_map = torch.zeros(3, device=pred["instance_per_point"].device)
            assign = assignment[b]
            instance_map[assign[1]] = assign[0].float()
            matched_gt_instance = instance_map[gt_instance_per_point.long()]

            loss_instance += loss.compute_miou_loss(pred["instance_per_point"][b, moving_mask].unsqueeze(0), F.one_hot(matched_gt_instance.long(), num_classes=self.max_K).unsqueeze(0))

            # Calculate the loss for the motion type
            loss_mtype += loss.compute_miou_loss(pred["mtype_per_point"][b, moving_mask].unsqueeze(0), F.one_hot(gt["mtype_per_point"][b, moving_mask].long(), num_classes=2).unsqueeze(0))

            # Calculate the loss for the motion axis, 2 is the non-joint category
            loss_maxis += loss.compute_vect_loss(pred["maxis_per_point"][b, moving_mask].unsqueeze(0), gt["maxis_per_point"][b, moving_mask].unsqueeze(0))

            # Calculate the loss for the motion origin, 0 is the rotation joint
            rot_mask = torch.logical_and(moving_mask, gt["mtype_per_point"][b, :] == 0)
            loss_morigin += loss.compute_vect_loss(pred["morigin_per_point"][b, rot_mask].unsqueeze(0), gt["morigin_per_point"][b, rot_mask].unsqueeze(0))

        loss_dict = {
            "loss_category": loss_category,
            "loss_instance": loss_instance / batch_size,
            "loss_mtype": loss_mtype / batch_size,
            "loss_maxis": loss_maxis / batch_size,
            "loss_morigin": loss_morigin / batch_size,
        }

        return loss_dict
