import math
import torch

def filter_boxes_opr(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    # keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    keep = (ws >= min_size) * (hs >= min_size)
    # NOTE: In FPN, I have met np.all(keep) = 0(don't know why),
    # thus I add the following line to avoid crash
    #keep = keep + (keep.sum().eq(0))
    return keep

def clip_boxes_opr(boxes, im_info):
    """ Clip the boxes into the image region."""
    w = im_info[1] - 1
    h = im_info[0] - 1
    boxes[:, 0::4] = boxes[:, 0::4].clamp(min=0, max=w)
    boxes[:, 1::4] = boxes[:, 1::4].clamp(min=0, max=h)
    boxes[:, 2::4] = boxes[:, 2::4].clamp(min=0, max=w)
    boxes[:, 3::4] = boxes[:, 3::4].clamp(min=0, max=h)
    return boxes

def bbox_transform_inv_opr(bbox, deltas):
    max_delta = math.log(1000.0 / 16)
    """ Transforms the learned deltas to the final bbox coordinates, the axis is 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height
    pred_ctr_x = bbox_ctr_x + deltas[:, 0] * bbox_width
    pred_ctr_y = bbox_ctr_y + deltas[:, 1] * bbox_height

    dw = deltas[:, 2]
    dh = deltas[:, 3]
    dw = torch.clamp(dw, max=max_delta)
    dh = torch.clamp(dh, max=max_delta)
    pred_width = bbox_width * torch.exp(dw)
    pred_height = bbox_height * torch.exp(dh)

    pred_x1 = pred_ctr_x - 0.5 * pred_width
    pred_y1 = pred_ctr_y - 0.5 * pred_height
    pred_x2 = pred_ctr_x + 0.5 * pred_width
    pred_y2 = pred_ctr_y + 0.5 * pred_height
    pred_boxes = torch.cat((pred_x1.reshape(-1, 1), pred_y1.reshape(-1, 1),
                            pred_x2.reshape(-1, 1), pred_y2.reshape(-1, 1)), dim=1)
    return pred_boxes

def bbox_transform_opr(bbox, gt):
    """ Transform the bounding box and ground truth to the loss targets.
    The 4 box coordinates are in axis 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1
    gt_height = gt[:, 3] - gt[:, 1] + 1
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = torch.log(gt_width / bbox_width)
    target_dh = torch.log(gt_height / bbox_height)
    target = torch.cat((target_dx.reshape(-1, 1), target_dy.reshape(-1, 1),
                        target_dw.reshape(-1, 1), target_dh.reshape(-1, 1)), dim=1)
    return target

def box_overlap_opr(box, gt):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 4) Tensor
    return: (N, K) Tensor, stores Max(0, intersection/union)
    """
    assert box.ndim == 2
    assert gt.ndim == 2
    #device = box.device
    #box = box.cpu()
    #gt = gt.cpu()
    N = box.shape[0]
    K = gt.shape[0]
    #target_shape = (N, K, 4)
    b_box = box.repeat(K,1)
    b_gt = gt.repeat(1,N).reshape(-1,4)
    iw = (
         torch.min(b_box[:, 2], b_gt[:, 2]) -
         torch.max(b_box[:, 0], b_gt[:, 0]) + 1).reshape(K, N).T.clamp(min=0)
    ih = (
         torch.min(b_box[:, 3], b_gt[:, 3]) -
         torch.max(b_box[:, 1], b_gt[:, 1]) + 1).reshape(K, N).T.clamp(min=0)
    inter = iw * ih
    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    #area_target_shape = (N, K)
    b_area_box = area_box.reshape(-1,1).repeat(1, K)
    b_area_gt = area_gt.repeat(N, 1)
    union = b_area_box + b_area_gt - inter
    overlaps = (inter / union).clamp(min=0)
    #overlaps = overlaps.cuda(device)
    return overlaps

def box_overlap_ignore_opr(box, gt, ignore_label=-1):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 4) Tensor
    return: (N, K) Tensor, stores Max(0, intersection/union)
    """
    assert box.ndim == 2
    assert gt.ndim == 2
    N = box.shape[0]
    K = gt.shape[0]
    b_box = box.repeat(K,1)
    b_gt = gt[:, :4].repeat(1,N).reshape(-1,4)
    iw = (
         torch.min(b_box[:, 2], b_gt[:, 2]) -
         torch.max(b_box[:, 0], b_gt[:, 0]) + 1).reshape(K, N).T.clamp(min=0)
    ih = (
         torch.min(b_box[:, 3], b_gt[:, 3]) -
         torch.max(b_box[:, 1], b_gt[:, 1]) + 1).reshape(K, N).T.clamp(min=0)
    inter = iw * ih
    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    b_area_box = area_box.reshape(-1,1).repeat(1, K)
    b_area_gt = area_gt.repeat(N, 1)
    union = b_area_box + b_area_gt - inter

    overlaps_normal = (inter / union).clamp(min=0)
    overlaps_ignore = (inter / b_area_box).clamp(min=0)

    gt_ignore_mask = gt[:, 4].eq(ignore_label).repeat(N, 1)
    overlaps_normal *= ~gt_ignore_mask
    overlaps_ignore *= gt_ignore_mask
    return overlaps_normal, overlaps_ignore

def box_overlap_ioa(box, gt):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 4) Tensor
    return: (N, K) Tensor, stores Max(0, intersection/union)
    """
    assert box.ndim == 2
    assert gt.ndim == 2
    #device = box.device
    N = box.shape[0]
    K = gt.shape[0]
    #target_shape = (N, K, 4)
    b_box = box.repeat(K,1)
    b_gt = gt.repeat(1,N).reshape(-1,4)
    iw = (
         torch.min(b_box[:, 2], b_gt[:, 2]) -
         torch.max(b_box[:, 0], b_gt[:, 0]) + 1).reshape(K, N).T.clamp(min=0)
    ih = (
         torch.min(b_box[:, 3], b_gt[:, 3]) -
         torch.max(b_box[:, 1], b_gt[:, 1]) + 1).reshape(K, N).T.clamp(min=0)
    inter = iw * ih
    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    #area_target_shape = (N, K)
    b_area_box = area_box.reshape(-1,1).repeat(1, K)
    overlaps = (inter / b_area_box).clamp(min=0)
    return overlaps

def box_overlap_ignore_ioa_forhead(box, gt, ignore_label=-1):
    """
    Compute the overlaps between box and gt(_box)
    box: (N, 4) Tensor
    gt : (K, 4) Tensor
    return: (N, K) Tensor, stores Max(0, intersection/union)
    """
    assert box.ndim == 2
    assert gt.ndim == 2
    N = box.shape[0]
    K = gt.shape[0]
    b_box = box.repeat(K,1)
    b_gt = gt[:, :4].repeat(1,N).reshape(-1,4)
    iw = (
         torch.min(b_box[:, 2], b_gt[:, 2]) -
         torch.max(b_box[:, 0], b_gt[:, 0]) + 1).reshape(K, N).T.clamp(min=0)
    ih = (
         torch.min(b_box[:, 3], b_gt[:, 3]) -
         torch.max(b_box[:, 1], b_gt[:, 1]) + 1).reshape(K, N).T.clamp(min=0)
    inter = iw * ih
    # Use the broadcast to save some time
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    b_area_box = area_box.reshape(-1,1).repeat(1, K)
    b_area_gt = area_gt.repeat(N, 1)
    #union = b_area_box + b_area_gt - inter

    overlaps_normal = (inter / b_area_gt).clamp(min=0)
    overlaps_ignore = (inter / b_area_box).clamp(min=0)

    gt_ignore_mask = gt[:, 4].eq(ignore_label).repeat(N, 1)
    overlaps_normal *= ~gt_ignore_mask
    overlaps_ignore *= gt_ignore_mask
    return overlaps_normal, overlaps_ignore

if __name__ == '__main__':
    a = torch.tensor(np.random.rand(3,4)*100)
    b = torch.tensor(np.random.rand(3,4)*100)
    print(box_overlap_opr(a,b))

