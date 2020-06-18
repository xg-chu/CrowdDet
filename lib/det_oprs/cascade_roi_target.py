# -*- coding: utf-8 -*-
import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr, box_overlap_ignore_opr

@torch.no_grad()
def cascade_roi_target(rpn_rois, im_info, gt_boxes, pos_threshold=0.5, top_k=1):
    return_rois = []
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        batch_inds = torch.ones((gt_boxes_perimg.shape[0], 1)).type_as(gt_boxes_perimg) * bid
        gt_rois = torch.cat([batch_inds, gt_boxes_perimg[:, :4]], axis=1)
        batch_roi_inds = torch.nonzero(rpn_rois[:, 0] == bid, as_tuple=False).flatten()
        all_rois = torch.cat([rpn_rois[batch_roi_inds], gt_rois], axis=0)
        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois[:, 1:5], gt_boxes_perimg)
        overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(descending=True, dim=1)
        overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(descending=True, dim=1)
        # gt max and indices, ignore max and indices
        max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
        gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
        max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()
        # cons masks
        ignore_assign_mask = (max_overlaps_normal < pos_threshold) * (
                max_overlaps_ignore > max_overlaps_normal)
        max_overlaps = max_overlaps_normal * ~ignore_assign_mask + \
                max_overlaps_ignore * ignore_assign_mask
        gt_assignment = gt_assignment_normal * ~ignore_assign_mask + \
                gt_assignment_ignore * ignore_assign_mask
        labels = gt_boxes_perimg[gt_assignment, 4]
        fg_mask = (max_overlaps >= pos_threshold) * (labels != config.ignore_label)
        bg_mask = (max_overlaps < config.bg_threshold_high)
        fg_mask = fg_mask.reshape(-1, top_k)
        bg_mask = bg_mask.reshape(-1, top_k)
        #pos_max = config.num_rois * config.fg_ratio
        #fg_inds_mask = subsample_masks(fg_mask[:, 0], pos_max, True)
        #neg_max = config.num_rois - fg_inds_mask.sum()
        #bg_inds_mask = subsample_masks(bg_mask[:, 0], neg_max, True)
        labels = labels * fg_mask.flatten()
        #keep_mask = fg_inds_mask + bg_inds_mask
        # labels
        labels = labels.reshape(-1, top_k)#[keep_mask]
        gt_assignment = gt_assignment.reshape(-1, top_k).flatten()
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        #rois = all_rois[keep_mask]
        target_rois = all_rois.repeat(1, top_k).reshape(-1, all_rois.shape[-1])
        bbox_targets = bbox_transform_opr(target_rois[:, 1:5], target_boxes)
        if config.rcnn_bbox_normalize_targets:
            std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
            mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
            minus_opr = mean_opr / std_opr
            bbox_targets = bbox_targets / std_opr - minus_opr
        bbox_targets = bbox_targets.reshape(-1, top_k * 4)
        return_rois.append(all_rois)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)
    if config.train_batch_per_gpu == 1:
        return rois, labels, bbox_targets
    else:
        return_rois = torch.cat(return_rois, axis=0)
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return return_rois, return_labels, return_bbox_targets

