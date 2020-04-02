#coding:utf-8
import numpy as np
from .matching import maxWeightMatching

def compute_matching(dt_boxes, gt_boxes, bm_thr):
    assert dt_boxes.shape[-1] > 3 and gt_boxes.shape[-1] > 3
    if dt_boxes.shape[0] < 1 or gt_boxes.shape[0] < 1:
        return list()
    N, K = dt_boxes.shape[0], gt_boxes.shape[0]
    ious = compute_iou_matrix(dt_boxes, gt_boxes)
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches

def compute_head_body_matching(dt_body, dt_head, gt_body, gt_head, bm_thr):
    assert dt_body.shape[-1] > 3 and gt_body.shape[-1] > 3
    assert dt_head.shape[-1] > 3 and gt_head.shape[-1] > 3
    assert dt_body.shape[0] == dt_head.shape[0]
    assert gt_body.shape[0] == gt_head.shape[0]
    N, K = dt_body.shape[0], gt_body.shape[0]
    ious_body = compute_iou_matrix(dt_body, gt_body)
    ious_head = compute_iou_matrix(dt_head, gt_head)
    mask_body = ious_body > bm_thr
    mask_head = ious_head > bm_thr
    # only keep the both matches detections
    mask = np.array(mask_body) & np.array(mask_head)
    ious = np.zeros((N, K))
    ious[mask] = (ious_body[mask] + ious_head[mask]) / 2
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches

def compute_multi_head_body_matching(dt_body, dt_head_0, dt_head_1, gt_body, gt_head, bm_thr):
    assert dt_body.shape[-1] > 3 and gt_body.shape[-1] > 3
    assert dt_head_0.shape[-1] > 3 and gt_head.shape[-1] > 3
    assert dt_head_1.shape[-1] > 3 and gt_head.shape[-1] > 3
    assert dt_body.shape[0] == dt_head_0.shape[0]
    assert gt_body.shape[0] == gt_head.shape[0]
    N, K = dt_body.shape[0], gt_body.shape[0]
    ious_body = compute_iou_matrix(dt_body, gt_body)
    ious_head_0 = compute_iou_matrix(dt_head_0, gt_head)
    ious_head_1 = compute_iou_matrix(dt_head_1, gt_head)
    mask_body = ious_body > bm_thr
    mask_head_0 = ious_head_0 > bm_thr
    mask_head_1 = ious_head_1 > bm_thr
    mask_head = mask_head_0 | mask_head_1
    # only keep the both matches detections
    mask = np.array(mask_body) & np.array(mask_head)
    ious = np.zeros((N, K))
    #ious[mask] = (ious_body[mask] + ious_head[mask]) / 2
    ious[mask] = ious_body[mask]
    rows, cols = np.where(ious > bm_thr)
    bipartites = [(i + 1, j + N + 1, ious[i, j]) for (i, j) in zip(rows, cols)]
    mates = maxWeightMatching(bipartites)
    if len(mates) < 1:
        return list()
    rows = np.where(np.array(mates) > -1)[0]
    indices = np.where(rows < N + 1)[0]
    rows = rows[indices]
    cols = np.array([mates[i] for i in rows])
    matches = [(i-1, j - N - 1) for (i, j) in zip(rows, cols)]
    return matches

def get_head_body_ignores(dt_body, dt_head, gt_body, gt_head, bm_thr):
    if gt_body.size:
        body_ioas = compute_ioa_matrix(dt_body, gt_body)
        head_ioas = compute_ioa_matrix(dt_head, gt_head)
        body_ioas = np.max(body_ioas, axis=1)
        head_ioas = np.max(head_ioas, axis=1)
        head_rows = np.where(head_ioas > bm_thr)[0]
        body_rows = np.where(body_ioas > bm_thr)[0]
        rows = set.union(set(head_rows), set(body_rows))
        return len(rows)
    else:
        return 0

def get_ignores(dt_boxes, gt_boxes, bm_thr):
    if gt_boxes.size:
        ioas = compute_ioa_matrix(dt_boxes, gt_boxes)
        ioas = np.max(ioas, axis = 1)
        rows = np.where(ioas > bm_thr)[0]
        return len(rows)
    else:
        return 0

def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):
    eps = 1e-6
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)
    ioas = inter / (dtarea + eps)
    return ioas

def compute_iou_matrix(dboxes:np.ndarray, gboxes:np.ndarray):
    eps = 1e-6
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
    ious = inter / (dtarea + gtarea - inter + eps)
    return ious

