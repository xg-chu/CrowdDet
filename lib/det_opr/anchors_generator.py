import torch
import numpy as np
from .anchors_generator_cell import generate_anchors


def generate_anchors_opr(fm_map, fm_stride, anchor_scales=(8, 16, 32),
                         anchor_ratios=(0.5, 1, 2), base_size=16):
    """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
    """
    np_anchors = generate_anchors(
        base_size=base_size,
        ratios=np.array(anchor_ratios),
        scales=np.array(anchor_scales))
    fm_height, fm_width = fm_map.shape[-2], fm_map.shape[-1]
    f_device = fm_map.device
    cell_anchors = torch.tensor(np_anchors, device=f_device)
    shift_x = torch.linspace(0, fm_width - 1, fm_width, device=f_device) * fm_stride
    shift_y = torch.linspace(0, fm_height - 1, fm_height, device=f_device) * fm_stride
    broad_shift_x = shift_x.reshape(-1, shift_x.shape[0]).repeat(fm_height,1)
    broad_shift_y = shift_y.reshape(shift_y.shape[0], -1).repeat(1,fm_width)

    flatten_shift_x = broad_shift_x.flatten().reshape(-1,1)
    flatten_shift_y = broad_shift_y.flatten().reshape(-1,1)

    shifts = torch.cat(
        [flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y],
        axis=1)
    all_anchors = shifts.repeat(1,3) + cell_anchors.flatten()
    all_anchors = all_anchors.reshape(-1, 4)
    # x1y1*3, x2y1*3, ... ,x1y2*3,x2y2*3, ... , xnyn*3
    return all_anchors

