import os
import sys
import math
import argparse
from multiprocessing import Queue, Process

from tqdm import tqdm
import numpy as np

from .JIToolkits.JI_tools import compute_matching, get_ignores
sys.path.insert(0, '../')
import utils.misc_utils as misc_utils

gtfile = '/data/annotation_val.odgt'
nr_procs = 10

def evaluation_all(path, target_key):
    records = misc_utils.load_json_lines(path)
    res_line = []
    res_JI = []
    for i in range(10):
        score_thr = 1e-1 * i
        total = len(records)
        stride = math.ceil(total / nr_procs)
        result_queue = Queue(10000)
        results, procs = [], []
        for i in range(nr_procs):
            start = i*stride
            end = np.min([start+stride,total])
            sample_data = records[start:end]
            p = Process(target= compute_JI_with_ignore, args=(result_queue, sample_data, score_thr, target_key))
            p.start()
            procs.append(p)
        tqdm.monitor_interval = 0
        pbar = tqdm(total=total, leave = False, ascii = True)
        for i in range(total):
            t = result_queue.get()
            results.append(t)
            pbar.update(1)
        for p in procs:
            p.join()
        pbar.close()
        line, mean_ratio = gather(results)
        line = 'score_thr:{:.1f}, {}'.format(score_thr, line)
        print(line)
        res_line.append(line)
        res_JI.append(mean_ratio)
    return res_line, max(res_JI)

def compute_JI_with_ignore(result_queue, records, score_thr, target_key, bm_thresh=0.5):
    for record in records:
        gt_boxes = misc_utils.load_bboxes(record, 'gtboxes', target_key, 'tag')
        gt_boxes[:,2:4] += gt_boxes[:,:2]
        gt_boxes = misc_utils.clip_boundary(gt_boxes, record['height'], record['width'])
        dt_boxes = misc_utils.load_bboxes(record, 'dtboxes', target_key, 'score')
        dt_boxes[:,2:4] += dt_boxes[:,:2]
        dt_boxes = misc_utils.clip_boundary(dt_boxes, record['height'], record['width'])
        keep = dt_boxes[:, -1] > score_thr
        dt_boxes = dt_boxes[keep][:, :-1]

        gt_tag = np.array(gt_boxes[:,-1]!=-1)
        matches = compute_matching(dt_boxes, gt_boxes[gt_tag, :4], bm_thresh)
        # get the unmatched_indices
        matched_indices = np.array([j for (j,_) in matches])
        unmatched_indices = list(set(np.arange(dt_boxes.shape[0])) - set(matched_indices))
        num_ignore_dt = get_ignores(dt_boxes[unmatched_indices], gt_boxes[~gt_tag, :4], bm_thresh)
        matched_indices = np.array([j for (_,j) in matches])
        unmatched_indices = list(set(np.arange(gt_boxes[gt_tag].shape[0])) - set(matched_indices))
        num_ignore_gt = get_ignores(gt_boxes[gt_tag][unmatched_indices], gt_boxes[~gt_tag, :4], bm_thresh)
        # compurte results
        eps = 1e-6
        k = len(matches)
        m = gt_tag.sum() - num_ignore_gt
        n = dt_boxes.shape[0] - num_ignore_dt
        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover
        result_dict = dict(ratio = ratio, recall = recall, cover = cover,
            noise = noise, k = k, m = m, n = n)
        result_queue.put_nowait(result_dict)

def gather(results):
    assert len(results)
    img_num = 0
    for result in results:
        if result['n'] != 0 or result['m'] != 0:
            img_num += 1
    mean_ratio = np.sum([rb['ratio'] for rb in results]) / img_num
    mean_cover = np.sum([rb['cover'] for rb in results]) / img_num
    mean_recall = np.sum([rb['recall'] for rb in results]) / img_num
    mean_noise = 1 - mean_cover
    valids = np.sum([rb['k'] for rb in results])
    total = np.sum([rb['n'] for rb in results])
    gtn = np.sum([rb['m'] for rb in results])

    #line = 'mean_ratio:{:.4f}, mean_cover:{:.4f}, mean_recall:{:.4f}, mean_noise:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
    #    mean_ratio, mean_cover, mean_recall, mean_noise, valids, total, gtn)
    line = 'mean_ratio:{:.4f}, valids:{}, total:{}, gtn:{}'.format(
        mean_ratio, valids, total, gtn)
    return line, mean_ratio

def common_process(func, cls_list, nr_procs):
    total = len(cls_list)
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = cls_list[start:end]
        p = Process(target= func,args=(result_queue, sample_data))
        p.start()
        procs.append(p)
    for i in range(total):
        t = result_queue.get()
        if t is None:
            continue
        results.append(t)
    for p in procs:
        p.join()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a json result file with iou match')
    parser.add_argument('--detfile', required=True, help='path of json result file to load')
    parser.add_argument('--target_key', required=True)
    args = parser.parse_args()
    evaluation_all(args.detfile, args.target_key)
