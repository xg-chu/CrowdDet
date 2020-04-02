import torch

from config import config
from det_opr.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, \
    filter_boxes_opr
import det_tools_cuda as dtc
nms = dtc.nms


def find_top_rpn_proposals(is_train, rpn_bbox_offsets_list, rpn_cls_prob_list,
        all_anchors_list, im_info):
    prev_nms_top_n = config.train_prev_nms_top_n \
        if is_train else config.test_prev_nms_top_n
    post_nms_top_n = config.train_post_nms_top_n \
        if is_train else config.test_post_nms_top_n
    batch_per_gpu = config.train_batch_per_gpu if is_train else 1
    nms_threshold = config.rpn_nms_threshold
    box_min_size = config.rpn_min_box_size
    bbox_normalize_targets = config.rpn_bbox_normalize_targets
    bbox_normalize_means = config.bbox_normalize_means
    bbox_normalize_stds = config.bbox_normalize_stds

    list_size = len(rpn_bbox_offsets_list)

    return_rois = []
    return_inds = []
    for bid in range(batch_per_gpu):
        batch_proposals_list = []
        batch_probs_list = []
        if list_size == 1:
            pass
        else:
            for l in range(list_size):
                # get proposals and probs
                offsets = rpn_bbox_offsets_list[l][bid] \
                    .permute(1, 2, 0).reshape(-1, 4)
                if bbox_normalize_targets:
                    std_opr = torch.tensor(config.bbox_normalize_stds[None, :]
                        ).type_as(bbox_targets)
                    mean_opr = torch.tensor(config.bbox_normalize_means[None, :]
                        ).type_as(bbox_targets)
                    pred_offsets = pred_offsets * std_opr
                    pred_offsets = pred_offsets + mean_opr
                all_anchors = all_anchors_list[l]
                proposals = bbox_transform_inv_opr(all_anchors, offsets)
                if config.anchor_within_border:
                    proposals = clip_boxes_opr(proposals, im_info[bid, :])
                probs = rpn_cls_prob_list[l][bid] \
                        .permute(1,2,0).flatten()
                # gather the proposals and probs
                batch_proposals_list.append(proposals)
                batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        batch_keep_index = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_index]
        batch_probs = batch_probs[batch_keep_index]
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        batch_proposals = batch_proposals[keep]
        batch_probs = batch_probs[keep]
        num_proposals = min(post_nms_top_n, batch_probs.shape[0])
        batch_proposals = batch_proposals[:num_proposals]
        batch_probs = batch_probs[:num_proposals]

        batch_inds = torch.ones(batch_proposals.shape[0], 1
                                ).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1).detach()
        return_rois.append(batch_rois)
        return_inds.append(batch_rois.shape[0])

    if batch_per_gpu == 1:
        return batch_rois, [batch_rois.shape[0]]
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        import numpy as np
        return_inds = np.cumsum(return_inds)
        return concated_rois, return_inds
