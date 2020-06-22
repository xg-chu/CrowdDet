import os
import sys

import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))

class Crowd_human:
    class_names = ['background', 'person']
    num_classes = len(class_names)
    root_folder = '/data/CrowdHuman'
    image_folder = '/data/CrowdHuman/images'
    train_source = os.path.join('/data/CrowdHuman/annotation_train.odgt')
    eval_source = os.path.join('/data/CrowdHuman/annotation_val.odgt')

class Config:
    output_dir = 'outputs'
    model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    init_weights = '/data/model/resnet50_fbaug.pth'

    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    train_image_short_size = 800
    train_image_max_size = 1400
    eval_resize = True
    eval_image_short_size = 800
    eval_image_max_size = 1400
    seed_dataprovider = 3
    train_source = Crowd_human.train_source
    eval_source = Crowd_human.eval_source
    image_folder = Crowd_human.image_folder
    class_names = Crowd_human.class_names
    num_classes = Crowd_human.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    backbone_freeze_at = 2
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 3.125e-4
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2

    warm_iter = 800
    max_epoch = 50
    lr_decay = [33, 43]
    nr_images_epoch = 15000
    log_dump_interval = 20

    # ----------test config---------- #
    test_layer_topk = 1000
    test_nms = 0.5
    test_nms_method = 'set_nms'
    visulize_threshold = 0.3
    pred_cls_threshold = 0.01

    # ----------dataset config---------- #
    nr_box_dim = 5
    max_boxes_of_image = 500

    # --------anchor generator config-------- #
    anchor_base_size = 32 # the minimize anchor size in the bigest feature map.
    anchor_base_scale = [2**0, 2**(1/3), 2**(2/3)]
    anchor_aspect_ratios = [1, 2, 3]
    num_cell_anchors = len(anchor_aspect_ratios) * len(anchor_base_scale)

    # ----------binding&training config---------- #
    smooth_l1_beta = 0.1
    negative_thresh = 0.4
    positive_thresh = 0.5
    allow_low_quality = True

config = Config()
