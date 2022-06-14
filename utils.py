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
from sklearn import metrics
from matplotlib.pyplot import figure
from matplotlib import colors


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
    # calculate AUC
    frame_auc = roc_auc_score(y_true=np.squeeze(
        labels, axis=0), y_score=np.squeeze(anomal_scores))

    return frame_auc


def plot_ROC(anomal_scores, labels, auc, log_dir, dataset_type, method, trained_model_using):
    # plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=np.squeeze(
        labels, axis=0), y_score=np.squeeze(anomal_scores))

    # create ROC curve
    plt.title('Receiver Operating Characteristic \nmethod: ' +
              method + ', dataset: ' + dataset_type +
              ', trained model used: ' + trained_model_using)
    plt.plot(fpr, tpr, 'b', label='ROC curve (AUC = %0.4f)' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', label='random predict')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    #plt.plot([0, 1], [1, 0], color='black', linewidth=1.5, linestyle='dashed')
    #plt.legend(loc='lower right')

    plt.savefig(os.path.join(log_dir, 'ROC.png'))


def plot_anomaly_scores(anomaly_score_total_list, labels, log_dir, dataset_type, method, trained_model_using):
    matrix = np.array([labels == 1])

    # Mask the False occurences in the numpy array as 'bad' data
    matrix = np.ma.masked_where(matrix == True, matrix)

    # Create a ListedColormap with only the color green specified
    cmap = colors.ListedColormap(['none'])

    # Use the `set_bad` property of `colormaps` to set all the 'bad' data to red
    cmap.set_bad(color='lavenderblush')
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 7)
    plt.title('Anomaly score/frame, method: ' +
              method + ', dataset: ' + dataset_type +
              ', trained model used: ' + trained_model_using)
    ax.pcolormesh(matrix, cmap=cmap, edgecolor='none', linestyle='-', lw=1)

    y = anomaly_score_total_list
    x = np.arange(0, len(y))
    plt.plot(x, y, color="steelblue", label="score/frame")
    plt.legend(loc='lower left')
    plt.ylabel('Score')
    plt.xlabel('Frames')
    plt.savefig(os.path.join(log_dir, 'anomaly_score.png'))


def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        sum_score = (alpha*list1[i]+(1-alpha)*list2[i])
        list_result.append(sum_score)

    return list_result
