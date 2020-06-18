import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr

@torch.no_grad()
def retina_anchor_target(anchors, gt_boxes, im_info, top_k=1):
    total_anchor = anchors.shape[0]
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        anchors = anchors.type_as(gt_boxes_perimg)
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])
        # gt max and indices
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)
        max_overlaps= max_overlaps.flatten()
        gt_assignment= gt_assignment.flatten()
        _, gt_assignment_for_gt = torch.max(overlaps, axis=0)
        del overlaps
        # cons labels
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)
        ignore_mask = (max_overlaps < config.positive_thresh) * (
                max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1
        # cons bbox targets
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        target_anchors = anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        bbox_targets = bbox_transform_opr(target_anchors, target_boxes)
        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            low_quality_bbox_targets = bbox_transform_opr(
                anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets
        labels = labels.reshape(-1, 1 * top_k)
        bbox_targets = bbox_targets.reshape(-1, 4 * top_k)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_labels, return_bbox_targets

