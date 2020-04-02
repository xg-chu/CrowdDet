import os
import math
import argparse

import torch
from torch.multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm

import network
import dataset
import misc_utils
from config import config

if_set_nms = True

def eval_all(args):
    # model_path
    saveDir = config.model_dir
    evalDir = config.eval_dir
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 
            'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)
    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)
    # load data
    records = misc_utils.load_json_lines(config.eval_source)
    # multiprocessing
    num_records = len(records)
    num_devs = len(devices)
    num_image = math.ceil(num_records / num_devs)
    result_queue = Queue(500)
    procs = []
    all_results = []
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, num_records)
        split_records = records[start:end]
        proc = Process(target=inference, args=(
                model_file, devices[i], split_records, result_queue))
        proc.start()
        procs.append(proc)
    pbar = tqdm(total=num_records, ncols=50)
    for i in range(num_records):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)
    for p in procs:
        p.join()
    fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    misc_utils.save_json_lines(all_results, fpath)

def inference(model_file, device, records, result_queue):
    torch.set_default_tensor_type('torch.FloatTensor')
    net = network.Network()
    net.cuda(device)
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])
    for record in records:
        np.set_printoptions(precision=2, suppress=True)
        net.eval()
        image, gt_boxes, im_info, ID = get_data(record, device)
        pred_boxes = net(image, im_info)
        if if_set_nms:
            from set_nms_utils import set_cpu_nms
            n = pred_boxes.shape[0] // 2
            idents = np.tile(np.arange(n)[:,None], (1, 2)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))
            keep = pred_boxes[:, -2] > 0.05
            pred_boxes = pred_boxes[keep]
            keep = set_cpu_nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]
        else:
            import det_tools_cuda as dtc
            nms = dtc.nms
            keep = nms(pred_boxes[:, :4], pred_boxes[:, 4], 0.5)
            pred_boxes = pred_boxes[keep]
            pred_boxes = np.array(pred_boxes)
            keep = pred_boxes[:, -1] > 0.05
            pred_boxes = pred_boxes[keep]
        result_dict = dict(ID=ID, height=int(im_info[0, -2]), width=int(im_info[0, -1]),
                dtboxes=boxes_dump(pred_boxes, False),
                gtboxes=boxes_dump(gt_boxes, True))
                #rois=misc_utils.boxes_dump(rois[:, 1:], True))
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes, is_gt):
    result = []
    boxes = boxes.tolist()
    for box in boxes:
        if is_gt:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
            result.append(box_dict)
        else:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = 1
            box_dict['proposal_num'] = box[-1]
            box_dict['score'] = box[-2]
            result.append(box_dict)
    return result

def get_data(record, device):
    data = dataset.val_dataset(record)
    image, gt_boxes, ID = \
                data['data'], data['boxes'], data['ID']
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    transposed_img = np.ascontiguousarray(
        resized_img.transpose(2, 0, 1)[None, :, :, :],
        dtype=np.float32)
    im_info = np.array([height, width, scale, original_height, original_width],
                       dtype=np.float32)
    image = torch.Tensor(transposed_img).cuda(device)
    im_info = torch.Tensor(im_info[None, :]).cuda(device)
    return image, gt_boxes, im_info, ID

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    args = parser.parse_args()
    eval_all(args)

if __name__ == '__main__':
    run_test()

