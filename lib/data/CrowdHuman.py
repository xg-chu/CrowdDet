import os
import cv2
import torch
import numpy as np

from utils import misc_utils

class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, config, if_train):
        if if_train:
            self.training = True
            source = config.train_source
            self.short_size = config.train_image_short_size
            self.max_size = config.train_image_max_size
        else:
            self.training = False
            source = config.eval_source
            self.short_size = config.eval_image_short_size
            self.max_size = config.eval_image_max_size
        self.records = misc_utils.load_json_lines(source)
        self.config = config

    def __getitem__(self, index):
        return self.load_record(self.records[index])

    def __len__(self):
        return len(self.records)

    def load_record(self, record):
        if self.training:
            if_flap = np.random.randint(2) == 1
        else:
            if_flap = False
        # image
        image_path = os.path.join(self.config.image_folder, record['ID']+'.png')
        image = misc_utils.load_img(image_path)
        image_h = image.shape[0]
        image_w = image.shape[1]
        if if_flap:
            image = cv2.flip(image, 1)
        if self.training:
            # ground_truth
            gtboxes = misc_utils.load_gt(record, 'gtboxes', 'fbox', self.config.class_names)
            keep = (gtboxes[:, 2]>=0) * (gtboxes[:, 3]>=0)
            gtboxes=gtboxes[keep, :]
            gtboxes[:, 2:4] += gtboxes[:, :2]
            if if_flap:
                gtboxes = flip_boxes(gtboxes, image_w)
            # im_info
            nr_gtboxes = gtboxes.shape[0]
            im_info = np.array([0, 0, 1, image_h, image_w, nr_gtboxes])
            return image, gtboxes, im_info
        else:
            # image
            t_height, t_width, scale = target_size(
                    image_h, image_w, self.short_size, self.max_size)
            # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
            resized_image = cv2.resize(image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
            resized_image = resized_image.transpose(2, 0, 1)
            image = torch.tensor(resized_image).float()
            gtboxes = misc_utils.load_gt(record, 'gtboxes', 'fbox', self.config.class_names)
            gtboxes[:, 2:4] += gtboxes[:, :2]
            gtboxes = torch.tensor(gtboxes)
            # im_info
            nr_gtboxes = gtboxes.shape[0]
            im_info = torch.tensor([t_height, t_width, scale, image_h, image_w, nr_gtboxes])
            return image, gtboxes, im_info, record['ID']

    def merge_batch(self, data):
        # image
        images = [it[0] for it in data]
        gt_boxes = [it[1] for it in data]
        im_info = np.array([it[2] for it in data])
        batch_height = np.max(im_info[:, 3])
        batch_width = np.max(im_info[:, 4])
        padded_images = [pad_image(
                im, batch_height, batch_width, self.config.image_mean) for im in images]
        t_height, t_width, scale = target_size(
                batch_height, batch_width, self.short_size, self.max_size)
        # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
        resized_images = np.array([cv2.resize(
                im, (t_width, t_height), interpolation=cv2.INTER_LINEAR) for im in padded_images])
        resized_images = resized_images.transpose(0, 3, 1, 2)
        images = torch.tensor(resized_images).float()
        # ground_truth
        ground_truth = []
        for it in gt_boxes:
            gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))
            it[:, 0:4] *= scale
            max_box = min(self.config.max_boxes_of_image, len(it))
            gt_padded[:max_box] = it[:max_box]
            ground_truth.append(gt_padded)
        ground_truth = torch.tensor(ground_truth).float()
        # im_info
        im_info[:, 0] = t_height
        im_info[:, 1] = t_width
        im_info[:, 2] = scale
        im_info = torch.tensor(im_info)
        if max(im_info[:, -1] < 2):
            return None, None, None
        else:
            return images, ground_truth, im_info

def target_size(height, width, short_size, max_size):
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    return t_height, t_width, scale

def flip_boxes(boxes, im_w):
    flip_boxes = boxes.copy()
    for i in range(flip_boxes.shape[0]):
        flip_boxes[i, 0] = im_w - boxes[i, 2] - 1
        flip_boxes[i, 2] = im_w - boxes[i, 0] - 1
    return flip_boxes

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
