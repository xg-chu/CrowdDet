import math

import torch
from torchvision.ops import roi_align

def assign_boxes_to_levels(rois, min_level, max_level, canonical_box_size=224, canonical_level=4):
    """
        rois (Tensor): A tensor of shape (N, 5).
        min_level (int), max_level (int), canonical_box_size (int), canonical_level (int).
        Return a tensor of length N.
    """
    eps = 1e-6
    box_sizes = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level

def roi_pooler(fpn_fms, rois, stride, pool_shape, pooler_type):
    if pooler_type == "ROIAlign":
        pooler_aligned = False
    elif pooler_type == "ROIAlignV2":
        pooler_aligned = True
    else:
        raise ValueError("Unknown pooler type: {}".format(pooler_type))
    assert len(fpn_fms) == len(stride)
    max_level = int(math.log2(stride[-1]))
    min_level = int(math.log2(stride[0]))
    assert (len(stride) == max_level - min_level + 1)
    level_assignments = assign_boxes_to_levels(rois, min_level, max_level, 224, 4)
    dtype, device = fpn_fms[0].dtype, fpn_fms[0].device
    output = torch.zeros((len(rois), fpn_fms[0].shape[1], pool_shape[0], pool_shape[1]),
            dtype=dtype, device=device)
    for level, (fm_level, scale_level) in enumerate(zip(fpn_fms, stride)):
        inds = torch.nonzero(level_assignments == level, as_tuple=False).squeeze(1)
        rois_level = rois[inds]
        output[inds] = roi_align(fm_level, rois_level, pool_shape, spatial_scale=1.0/scale_level,
                sampling_ratio=-1, aligned=pooler_aligned)
    return output

