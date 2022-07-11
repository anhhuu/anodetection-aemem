import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import copy
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from matplotlib import colors
import cv2

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
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), 
                              y_score=np.squeeze(anomal_scores))

    return frame_auc


def plot_ROC(anomal_scores, labels, auc, log_dir, dataset_type, method, trained_model_using):
    # plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true=np.squeeze(labels, axis=0), 
                                    y_score=np.squeeze(anomal_scores))

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


def load_pixelLabel_frames(dataset_type='ped2'):
    label_input_path = []
    label_dir = []
    label_dir_distinct = []
    cur_path = './dataset/' + dataset_type + '/testing/labels'
    for path, _, files in os.walk(cur_path):
        for name in files:
            if(path not in label_dir_distinct):
                label_dir_distinct.append(path)
            label_input_path.append(os.path.join(path, name))
            label_dir.append(path)
    label_input_path.sort()
    label_dir.sort()
    label_dir_distinct.sort()

    label_list = []
    for i in range(len(label_input_path)):
        label_img = cv2.imread(label_input_path[i])
        label_list.append(label_img)

    return label_list


def load_predict_frames(dataset_type):
    pred_input_path = []
    cur_path = './dataset/' + dataset_type + '/output/frames'
    for path, _, files in os.walk(cur_path):
        for name in files:
            pred_input_path.append(os.path.join(path, name))
    pred_input_path.sort()

    pred_input_imgs = []
    for i in range(len(pred_input_path)):
        img = cv2.imread(pred_input_path[i])
        pred_input_imgs.append(img)

    return pred_input_imgs


def AUC_pixel_level():
    labels_frames = load_pixelLabel_frames(dataset_type='ped2')
    predicted_frames = load_predict_frames(dataset_type='ped2')
    


def optimal_threshold(anomal_scores, labels):
    y_true = 1 - labels
    y_score = np.squeeze(anomal_scores)
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    frame_auc = metrics.roc_auc_score(y_true, y_score)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
    return threshold[ix]



def average_score(anomaly_score, opt_threshold):
    count_nomaly = 0
    sum_nomaly = 0
    count_anomaly = 0
    sum_anomaly = 0
    for i in range(len(anomaly_score)):
        if anomaly_score[i] < opt_threshold:
            sum_anomaly += anomaly_score[i]
            count_anomaly += 1
        else:
            sum_nomaly += anomaly_score[i]
            count_nomaly += 1
    return sum_nomaly/count_nomaly, sum_anomaly/count_anomaly
