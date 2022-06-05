import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def psnr(mse):
    psnr_score = 10 * math.log10(1 / mse)
    return psnr_score


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):
    img_re = copy.copy(img)
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))

    return img_re


def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(
        normal*loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score


def anomaly_score(psnr, max_psnr, min_psnr):
    min_max_score = ((psnr - min_psnr) / (max_psnr-min_psnr))
    return min_max_score


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    inv_min_max_score = (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))
    return inv_min_max_score


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        min_max_score = anomaly_score(
            psnr_list[i], np.max(psnr_list), np.min(psnr_list))
        anomaly_score_list.append(min_max_score)

    return anomaly_score_list


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        min_max_score_inv = anomaly_score_inv(
            psnr_list[i], np.max(psnr_list), np.min(psnr_list))
        anomaly_score_list.append(min_max_score_inv)

    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(
        labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        sum_score = (alpha*list1[i]+(1-alpha)*list2[i])
        list_result.append(sum_score)

    return list_result
