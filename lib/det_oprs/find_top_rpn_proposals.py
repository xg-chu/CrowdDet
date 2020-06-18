import torch
import torch.nn.functional as F

from config import config
from det_oprs.bbox_opr import bbox_transform_inv_opr, clip_boxes_opr, \
    filter_boxes_opr
from torchvision.ops import nms

@torch.no_grad()
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
        for l in range(list_size):
            # get proposals and probs
            offsets = rpn_bbox_offsets_list[l][bid] \
                .permute(1, 2, 0).reshape(-1, 4)
            if bbox_normalize_targets:
                std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
                mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
                pred_offsets = pred_offsets * std_opr
                pred_offsets = pred_offsets + mean_opr
            all_anchors = all_anchors_list[l]
            proposals = bbox_transform_inv_opr(all_anchors, offsets)
            if config.anchor_within_border:
                proposals = clip_boxes_opr(proposals, im_info[bid, :])
            probs = rpn_cls_prob_list[l][bid] \
                    .permute(1,2,0).reshape(-1, 2)
            probs = torch.softmax(probs, dim=-1)[:, 1]
            # gather the proposals and probs
            batch_proposals_list.append(proposals)
            batch_probs_list.append(probs)
        batch_proposals = torch.cat(batch_proposals_list, dim=0)
        batch_probs = torch.cat(batch_probs_list, dim=0)
        # filter the zero boxes.
        batch_keep_mask = filter_boxes_opr(
                batch_proposals, box_min_size * im_info[bid, 2])
        batch_proposals = batch_proposals[batch_keep_mask]
        batch_probs = batch_probs[batch_keep_mask]
        # prev_nms_top_n
        num_proposals = min(prev_nms_top_n, batch_probs.shape[0])
        batch_probs, idx = batch_probs.sort(descending=True)
        batch_probs = batch_probs[:num_proposals]
        topk_idx = idx[:num_proposals].flatten()
        batch_proposals = batch_proposals[topk_idx]
        # For each image, run a total-level NMS, and choose topk results.
        keep = nms(batch_proposals, batch_probs, nms_threshold)
        keep = keep[:post_nms_top_n]
        batch_proposals = batch_proposals[keep]
        #batch_probs = batch_probs[keep]
        # cons the rois
        batch_inds = torch.ones(batch_proposals.shape[0], 1).type_as(batch_proposals) * bid
        batch_rois = torch.cat([batch_inds, batch_proposals], axis=1)
        return_rois.append(batch_rois)

    if batch_per_gpu == 1:
        return batch_rois
    else:
        concated_rois = torch.cat(return_rois, axis=0)
        return concated_rois
