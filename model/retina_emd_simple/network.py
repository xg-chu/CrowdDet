import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from det_oprs.anchors_generator import AnchorGenerator
from det_oprs.retina_anchor_target import retina_anchor_target
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.loss_opr import emd_loss_focal
from det_oprs.utils import get_padded_tensor

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.R_Head = RetinaNet_Head()
        self.R_Anchor = RetinaNet_Anchor()
        self.R_Criteria = RetinaNet_Criteria()

    def forward(self, image, im_info, gt_boxes=None):
        # pre-processing the data
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        # do inference
        # stride: 128,64,32,16,8, p7->p3
        fpn_fms = self.FPN(image)
        anchors_list = self.R_Anchor(fpn_fms)
        pred_cls_list, pred_reg_list = self.R_Head(fpn_fms)
        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(
                    pred_cls_list, pred_reg_list, anchors_list, gt_boxes, im_info)
            return loss_dict
        else:
            #pred_bbox = union_inference(
            #        anchors_list, pred_cls_list, pred_reg_list, im_info)
            pred_bbox = per_layer_inference(
                    anchors_list, pred_cls_list, pred_reg_list, im_info)
            return pred_bbox.cpu().detach()

class RetinaNet_Anchor():
    def __init__(self):
        self.anchors_generator = AnchorGenerator(
            config.anchor_base_size,
            config.anchor_aspect_ratios,
            config.anchor_base_scale)

    def __call__(self, fpn_fms):
        # get anchors
        all_anchors_list = []
        base_stride = 8
        off_stride = 2**(len(fpn_fms)-1) # 16
        for fm in fpn_fms:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)
        return all_anchors_list

class RetinaNet_Criteria(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_normalizer = 100 # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def __call__(self, pred_cls_list, pred_reg_list, anchors_list, gt_boxes, im_info):
        all_anchors = torch.cat(anchors_list, axis=0)
        all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, (config.num_classes-1)*2)
        all_pred_cls = torch.sigmoid(all_pred_cls)
        all_pred_reg = torch.cat(pred_reg_list, axis=1).reshape(-1, 4*2)
        # get ground truth
        labels, bbox_targets = retina_anchor_target(all_anchors, gt_boxes, im_info, top_k=2)
        all_pred_cls = all_pred_cls.reshape(-1, 2, config.num_classes-1)
        all_pred_reg = all_pred_reg.reshape(-1, 2, 4)
        loss0 = emd_loss_focal(
                all_pred_reg[:, 0], all_pred_cls[:, 0],
                all_pred_reg[:, 1], all_pred_cls[:, 1],
                bbox_targets, labels)
        loss1 = emd_loss_focal(
                all_pred_reg[:, 1], all_pred_cls[:, 1],
                all_pred_reg[:, 0], all_pred_cls[:, 0],
                bbox_targets, labels)
        del all_anchors
        del all_pred_cls
        del all_pred_reg
        loss = torch.cat([loss0, loss1], axis=1)
        # requires_grad = False
        _, min_indices = loss.min(axis=1)
        loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
        # only main labels
        num_pos = (labels[:, 0] > 0).sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum) * max(num_pos, 1)
        loss_emd = loss_emd.sum() / self.loss_normalizer
        loss_dict = {}
        loss_dict['retina_emd'] = loss_emd
        return loss_dict

class RetinaNet_Head(nn.Module):
    def __init__(self):
        super().__init__()
        num_convs = 4
        in_channels = 256
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        # predictor
        self.cls_score = nn.Conv2d(
            in_channels, config.num_cell_anchors * (config.num_classes-1) * 2,
            kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, config.num_cell_anchors * 4 * 2,
            kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet,
                self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        prior_prob = 0.01
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        pred_cls = []
        pred_reg = []
        for feature in features:
            pred_cls.append(self.cls_score(self.cls_subnet(feature)))
            pred_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        # reshape the predictions
        assert pred_cls[0].dim() == 4
        pred_cls_list = [
            _.permute(0, 2, 3, 1).reshape(pred_cls[0].shape[0], -1, (config.num_classes-1)*2)
            for _ in pred_cls]
        pred_reg_list = [
            _.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 4*2)
            for _ in pred_reg]
        return pred_cls_list, pred_reg_list

def per_layer_inference(anchors_list, pred_cls_list, pred_reg_list, im_info):
    keep_anchors = []
    keep_cls = []
    keep_reg = []
    class_num = pred_cls_list[0].shape[-1] // 2
    for l_id in range(len(anchors_list)):
        anchors = anchors_list[l_id].reshape(-1, 4)
        pred_cls = pred_cls_list[l_id][0].reshape(-1, class_num*2)
        pred_reg = pred_reg_list[l_id][0].reshape(-1, 4*2)
        if len(anchors) > config.test_layer_topk:
            ruler = pred_cls.max(axis=1)[0]
            _, inds = ruler.topk(config.test_layer_topk, dim=0)
            inds = inds.flatten()
            keep_anchors.append(anchors[inds])
            keep_cls.append(torch.sigmoid(pred_cls[inds]))
            keep_reg.append(pred_reg[inds])
        else:
            keep_anchors.append(anchors)
            keep_cls.append(torch.sigmoid(pred_cls))
            keep_reg.append(pred_reg)
    keep_anchors = torch.cat(keep_anchors, axis = 0)
    keep_cls = torch.cat(keep_cls, axis = 0)
    keep_reg = torch.cat(keep_reg, axis = 0)
    # multiclass
    tag = torch.arange(class_num).type_as(keep_cls)+1
    tag = tag.repeat(keep_cls.shape[0], 1).reshape(-1,1)
    pred_scores_0 = keep_cls[:, :class_num].reshape(-1, 1)
    pred_scores_1 = keep_cls[:, class_num:].reshape(-1, 1)
    pred_delta_0 = keep_reg[:, :4]
    pred_delta_1 = keep_reg[:, 4:]
    pred_bbox_0 = restore_bbox(keep_anchors, pred_delta_0, False)
    pred_bbox_1 = restore_bbox(keep_anchors, pred_delta_1, False)
    pred_bbox_0 = pred_bbox_0.repeat(1, class_num).reshape(-1, 4)
    pred_bbox_1 = pred_bbox_1.repeat(1, class_num).reshape(-1, 4)
    pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
    pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)
    pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)
    return pred_bbox

def union_inference(anchors_list, pred_cls_list, pred_reg_list, im_info):
    anchors = torch.cat(anchors_list, axis = 0)
    pred_cls = torch.cat(pred_cls_list, axis = 1)[0]
    pred_cls = torch.sigmoid(pred_cls)
    pred_reg = torch.cat(pred_reg_list, axis = 1)[0]
    class_num = pred_cls.shape[-1] // 2
    # multiclass
    tag = torch.arange(class_num).type_as(pred_cls)+1
    tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1,1)
    pred_scores_0 = pred_cls[:, :class_num].reshape(-1, 1)
    pred_scores_1 = pred_cls[:, class_num:].reshape(-1, 1)
    pred_delta_0 = pred_reg[:, :4]
    pred_delta_1 = pred_reg[:, 4:]
    pred_bbox_0 = restore_bbox(anchors, pred_delta_0, False)
    pred_bbox_1 = restore_bbox(anchors, pred_delta_1, False)
    pred_bbox_0 = pred_bbox_0.repeat(1, class_num).reshape(-1, 4)
    pred_bbox_1 = pred_bbox_1.repeat(1, class_num).reshape(-1, 4)
    pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
    pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)
    pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)
    return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

