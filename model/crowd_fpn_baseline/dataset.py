# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import threading

import cv2
import numpy as np

from config import config
import misc_utils

def train_dataset(seed=config.seed_dataprovider):
    root = config.image_folder
    source = config.train_source
    batch_per_gpu = config.train_batch_per_gpu
    short_size = config.train_image_short_size
    max_size = config.train_image_max_size
    records = misc_utils.load_json_lines(source)
    nr_files = len(records)
    print('training image number: {}'.format(nr_files))
    np.random.seed(seed)
    np.random.shuffle(records)
    file_idx = 0
    while file_idx < nr_files:
        batch_records = []
        batch_images_list = []
        hw_stat = np.zeros((batch_per_gpu, 2), np.int32)
        for i in range(batch_per_gpu):
            record = records[file_idx]
            batch_records.append(record)
            image_path = os.path.join(config.image_folder,record['ID'] + '.png')
            img = misc_utils.load_img(image_path)
            batch_images_list.append(img.copy())
            hw_stat[i, :] = img.shape[:2]
            file_idx += 1
            if file_idx >= nr_files:
                file_idx = 0
                np.random.shuffle(records)

        batch_image_height = np.max(hw_stat[:, 0])
        batch_image_width = np.max(hw_stat[:, 1])
        is_batch_ok = True
        batch_resized_height, batch_resized_width = get_hw_by_short_size(
            batch_image_height, batch_image_width, short_size, max_size)
        batch_images = np.zeros(
            (batch_per_gpu, 3, max_size, max_size),
            dtype=np.float32)
        batch_gts = np.zeros(
            (batch_per_gpu, config.max_boxes_of_image, config.nr_box_dim),
            dtype=np.float32)
        batch_info = np.zeros((batch_per_gpu, 6), dtype=np.float32)

        for i in range(batch_per_gpu):
            record = batch_records[i]
            img = batch_images_list[i]
            gt_boxes = misc_utils.load_gt(record, 'gtboxes', 'fbox', config.class_names)
            keep = (gt_boxes[:, 2]>=0) * (gt_boxes[:, 3]>=0)
            gt_boxes=gt_boxes[keep, :]
            nr_gtboxes = gt_boxes.shape[0]
            if nr_gtboxes == 0:
                is_batch_ok = False
                break
            gt_boxes[:, 2:4] += gt_boxes[:, :2]
            padded_image = pad_image(img, batch_image_height, 
                    batch_image_width, config.image_mean)
            original_height, original_width, channels = padded_image.shape
            resized_image, scale = resize_img_by_short_and_max_size(
                    padded_image, short_size, max_size)
            gt_boxes[:, 0:4] *= scale
            resized_gt = gt_boxes
            if np.random.randint(2) == 1:
                resized_image, resized_gt = flip_image_and_boxes(
                    resized_image, resized_gt)
            resized_image = resized_image.transpose(2, 0, 1).astype(np.float32)
            batch_images[i,:, :int(resized_image.shape[1]), :int(resized_image.shape[2])] = resized_image
            batch_gts[i, :nr_gtboxes] = resized_gt
            batch_info[i, :] = (
                resized_image.shape[1], resized_image.shape[2], scale,
                original_height, original_width, nr_gtboxes)
        if not is_batch_ok:
            continue
        yield dict(data=batch_images, boxes=batch_gts, im_info=batch_info)

def val_dataset(record):
    image_id = record['ID']
    gtboxes = record['gtboxes']
    image_path = os.path.join(config.image_folder, image_id + '.png')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gt_boxes = misc_utils.load_gt(record, 'gtboxes', 'fbox', config.class_names)
    gt_boxes[:, 2:4] += gt_boxes[:, :2]
    return dict(data=img, boxes = gt_boxes, ID = image_id)

def get_hw_by_short_size(im_height, im_width, short_size, max_size):
    im_size_min = np.min([im_height, im_width])
    im_size_max = np.max([im_height, im_width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max

    resized_height, resized_width = int(round(im_height * scale)), int(
        round(im_width * scale))
    return resized_height, resized_width

def resize_img_by_short_and_max_size(
        img, short_size, max_size, *, random_scale_methods=False):
    resized_height, resized_width = get_hw_by_short_size(
        img.shape[0], img.shape[1], short_size, max_size)
    scale = resized_height / (img.shape[0] + 0.0)

    chosen_resize_option = cv2.INTER_LINEAR
    if random_scale_methods:
        resize_options = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST,
                          cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    img = cv2.resize(img, (resized_width, resized_height),
                     interpolation=chosen_resize_option)
    return img, scale

def flip_image_and_boxes(img, boxes=None, *, segs=None):
    h, w, c = img.shape
    flip_img = cv2.flip(img, 1)
    if segs is not None:
        flip_segs = segs[:, :, ::-1]

    if boxes is not None:
        flip_boxes = boxes.copy()
        for i in range(flip_boxes.shape[0]):
            flip_boxes[i, 0] = w - boxes[i, 2] - 1  # x
            flip_boxes[i, 2] = w - boxes[i, 0] - 1  # x1
        if segs is not None:
            return flip_img, flip_boxes, flip_segs
        else:
            return flip_img, flip_boxes
    else:
        if segs is not None:
            return flip_img, flip_segs
        else:
            return flip_img

def pad_image(img, height, width, mean_value):
    o_h, o_w, _ = img.shape
    margins = np.zeros(2, np.int32)
    assert o_h <= height
    margins[0] = height - o_h
    img = cv2.copyMakeBorder(
        img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    img[o_h:, :, :] = mean_value

    assert o_w <= width
    margins[1] = width - o_w
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
    img[:, o_w:, :] = mean_value

    return img

