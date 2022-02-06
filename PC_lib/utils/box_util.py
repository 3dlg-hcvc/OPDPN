# box util code is modified based on 3DERT project https://github.com/facebookresearch/3detr/tree/d8a09c7dd27a0203116b960581c7c9d18a6910ac
import torch
import numpy as np

def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt(
        (corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    b = torch.sqrt(
        (corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    c = torch.sqrt(
        (corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS)
    )
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols

def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    EPS = 1e-6

    corners1 = corners1.clone()
    corners2 = corners2.clone()
    # flip Y axis, since it is negative
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1

    al_xmin = torch.min(
        torch.min(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymin = torch.max(
        torch.max(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmin = torch.min(
        torch.min(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )
    al_xmax = torch.max(
        torch.max(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymax = torch.min(
        torch.min(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmax = torch.max(
        torch.max(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )

    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol


def generalized_box3d_iou_tensor(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k1: torch.Tensor,
    nums_k2: torch.Tensor
):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8

    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    for b in range(B):
        non_rot_inter_areas[b, nums_k1[b] :, nums_k2[b] :] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)

    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)

    inter_areas = non_rot_inter_areas

    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious[torch.logical_not(good_boxes)] = 0

    mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
    for b in range(B):
        mask[b, : nums_k1[b], : nums_k2[b]] = 1
    gious = gious * mask
    return gious


# generalized_box3d_iou_tensor_jit = torch.jit.script(generalized_box3d_iou_tensor)

def generalized_box3d_iou(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k1: torch.Tensor,
    nums_k2: torch.Tensor
):
    return generalized_box3d_iou_tensor(
        corners1, corners2, nums_k1, nums_k2
    )

# Get the eight corners of the bbx
# box_size: B * K * 3; center: B * K * 3 
def get_corners_from_bbx(
    box_size: torch.Tensor,
    center: torch.Tensor,
):
    input_shape = center.shape[:-1]
    l = torch.unsqueeze(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)
    
    corners_3d = torch.zeros(tuple(list(input_shape) + [8, 3]), device=center.device)
    corners_3d[..., :, 0] = torch.cat(
        (l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1
    )
    corners_3d[..., :, 1] = torch.cat(
        (h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1
    )
    corners_3d[..., :, 2] = torch.cat(
        (w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1
    )
    corners_3d += torch.unsqueeze(center, -2)
    return corners_3d