import torch
import numpy as np

from det_opr.bbox_opr import box_overlap_opr, bbox_transform_opr
from config import config

def fpn_rpn_reshape(pred_cls_score_list, pred_bbox_offsets_list):
    final_pred_bbox_offsets_list = []
    final_pred_cls_score_list = []
    for bid in range(config.train_batch_per_gpu):
        batch_pred_bbox_offsets_list = []
        batch_pred_cls_score_list = []
        for i in range(config.num_anchor_scales):
            pred_cls_score_perlvl = pred_cls_score_list[i][bid] \
                .permute(1, 2, 0).flatten()
            pred_bbox_offsets_perlvl = pred_bbox_offsets_list[i][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            batch_pred_cls_score_list.append(pred_cls_score_perlvl)
            batch_pred_bbox_offsets_list.append(pred_bbox_offsets_perlvl)
        batch_pred_cls_score = torch.cat(batch_pred_cls_score_list, dim=0)
        batch_pred_bbox_offsets = torch.cat(batch_pred_bbox_offsets_list, dim=0)
        final_pred_cls_score_list.append(batch_pred_cls_score)
        final_pred_bbox_offsets_list.append(batch_pred_bbox_offsets)
    final_pred_cls_score = torch.cat(final_pred_cls_score_list, dim=0)
    final_pred_bbox_offsets = torch.cat(final_pred_bbox_offsets_list, dim=0)
    return final_pred_cls_score, final_pred_bbox_offsets

def fpn_anchor_target_opr_core_impl(
        gt_boxes, im_info, anchors, allow_low_quality_matches=True):
    ignore_label = config.ignore_label
    # get the gt boxes
    valid_gt_boxes = gt_boxes[:int(im_info[5]), :]
    valid_gt_boxes = valid_gt_boxes[valid_gt_boxes[:, -1].gt(0)]
    # compute the iou matrix
    overlaps = box_overlap_opr(anchors, valid_gt_boxes[:, :4])
    # match the dtboxes
    max_overlaps, argmax_overlaps = torch.max(overlaps, axis=1)
    # all ignore
    labels = torch.ones(anchors.shape[0], device=gt_boxes.device, dtype=torch.long) * ignore_label
    fg_mask = (max_overlaps >= config.rpn_positive_overlap)
    if allow_low_quality_matches:
        gt_max_overlaps, gt_argmax_overlaps = torch.max(overlaps, axis=0)
        ruler = ~fg_mask[gt_argmax_overlaps.long()]
        fg_mask[gt_argmax_overlaps.long()] = True
        gt_id = torch.arange(valid_gt_boxes.shape[0]).type_as(argmax_overlaps)
        det_id = gt_argmax_overlaps[ruler]
        argmax_overlaps[det_id] = gt_id[ruler]
        max_overlaps[det_id] = 0.5
    fg_mask_ind = torch.nonzero(fg_mask).flatten()
    # set positive ones
    labels[fg_mask_ind] = 1
    # set negative ones
    labels = labels * (max_overlaps >= config.rpn_negative_overlap)
    # bbox targets
    bbox_targets = bbox_transform_opr(
            anchors, valid_gt_boxes[argmax_overlaps, :4])
    if config.rpn_bbox_normalize_targets:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]
                ).type_as(bbox_targets)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]
                ).type_as(bbox_targets)
        minus_opr = mean_opr / std_opr
        bbox_targets = bbox_targets / std_opr - minus_opr
    return labels, bbox_targets

def fpn_anchor_target(boxes, im_info, all_anchors_list):
    final_labels_list = []
    final_bbox_targets_list = []
    for bid in range(config.train_batch_per_gpu):
        batch_labels_list = []
        batch_bbox_targets_list = []
        for i in range(config.num_anchor_scales):
            anchors_perlvl = all_anchors_list[i]
            rpn_labels_perlvl, rpn_bbox_targets_perlvl = fpn_anchor_target_opr_core_impl(
                boxes[bid], im_info[bid], anchors_perlvl)
            batch_labels_list.append(rpn_labels_perlvl)
            batch_bbox_targets_list.append(rpn_bbox_targets_perlvl)
        # here we samples the rpn_labels
        concated_batch_labels = torch.cat(batch_labels_list, dim=0)
        concated_batch_bbox_targets = torch.cat(batch_bbox_targets_list, dim=0)
        # sample labels
        pos_idx, neg_idx = subsample_labels(concated_batch_labels,
            config.num_sample_anchors, config.positive_anchor_ratio)
        concated_batch_labels.fill_(-1)
        concated_batch_labels.scatter_(0, pos_idx, 1)
        concated_batch_labels.scatter_(0, neg_idx, 0)

        final_labels_list.append(concated_batch_labels)
        final_bbox_targets_list.append(concated_batch_bbox_targets)
    final_labels = torch.cat(final_labels_list, dim=0)
    final_bbox_targets = torch.cat(final_bbox_targets_list, dim=0)
    return final_labels, final_bbox_targets

def subsample_labels(labels, num_samples, positive_fraction):
    positive = torch.nonzero((labels != config.ignore_label) & (labels != 0)).squeeze(1)
    negative = torch.nonzero(labels == 0).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx
