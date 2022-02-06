import torch.nn.functional as F
import torch

def compute_miou_loss(pred_seg_per_point, gt_seg_onehot):
    dot = torch.sum(pred_seg_per_point * gt_seg_onehot, axis=1)
    denominator = torch.sum(pred_seg_per_point, axis=1) + torch.sum(gt_seg_onehot, axis=1) - dot
    mIoU = dot / (denominator + 1e-10)
    return torch.mean(1.0 - mIoU)

def compute_vect_loss(pred_vect_per_point, gt_vect_per_point):
    diff_l2 = torch.norm(pred_vect_per_point - gt_vect_per_point, dim=2)

    return torch.mean(torch.mean(diff_l2, axis=1), axis=0)