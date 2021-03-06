import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from utils import *
import glob
import argparse
import cv2
from datetime import datetime

parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--test_batch_size', type=int,
                    default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256,
                    help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='weight for the anomality score')
parser.add_argument('--recon_alpha', type=float, default=0.7,
                    help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01,
                    help='threshold for test updating')
parser.add_argument('--recon_th', type=float, default=0.015,
                    help='threshold for test updating')
parser.add_argument('--num_workers_test', type=int, default=1,
                    help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2',
                    help='type of dataset: ped1, ped2, avenue')
parser.add_argument('--dataset_path', type=str,
                    default='./dataset', help='directory of data')
parser.add_argument('--pred_model_dir', type=str,
                    default='./pre_trained_model/defaults/ped2_prediction_model.pth', help='directory of model')
parser.add_argument('--pred_m_items_dir', type=str,
                    default='./pre_trained_model/defaults/ped2_prediction_keys.pt', help='directory of model')
parser.add_argument('--recon_model_dir', type=str,
                    default='./pre_trained_model/recon/ped2_reconstruction_model.pth', help='directory of model')
parser.add_argument('--recon_m_items_dir', type=str,
                    default='./pre_trained_model/recon/ped2_reconstruction_keys.pt', help='directory of model')
parser.add_argument('--exp_dir', type=str, default='log',
                    help='directory of log')
parser.add_argument('--is_save_output', type=str, default='false',
                    help='is save predicted image')

start_time = datetime.now()
print("Start time:", start_time.strftime("%d/%m/%Y %H:%M:%S"))

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

# make sure to use cudnn for computational performance
torch.backends.cudnn.enabled = True

test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

# load the dataset and convert a ndarray image/frame into a float tensor. Then scale the image/frame
# pixel intensity value in the range [-1, 1]
test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor(), ]),
                          resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
# dataset length
test_size = len(test_dataset)

# load data into mini test batch with batch_size:
#   + test_dataset: loader dataset
#   + batch_size: size of mini batch
#   + shuffle: not shuffle due to sequential data
#   + num_workers: how many subprocesses to use for data loading
#   + drop_last: If the size of dataset is not divisible by the batch size, then the last batch will be smaller
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# define Mean Error Loss
loss_func_mse = nn.MSELoss(reduction='none')

# Loading the pred trained model
pred_model = torch.load(args.pred_model_dir)
pred_model.cuda()
pred_m_items = torch.load(args.pred_m_items_dir)

# Loading the recon trained model
recon_model = torch.load(args.recon_model_dir)
recon_model.cuda()
recon_m_items = torch.load(args.recon_m_items_dir)

# load labels file of dataset
labels = np.load('./data_labels/frame_labels_'+args.dataset_type+'.npy')

# Setup a list contain video segments, element in the list contain all frames of this video segment.
videos = OrderedDict()  # './dataset/ped2/testing/frames/01'; .../02; ...
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

# initialize label list, psnr dict and feature_distance_list (compactness loss) dict
labels_list_full = []
labels_list_pred = []
label_length = 0
psnr_list = {}
recon_psnr_list = {}
feature_distance_list = {}
recon_feature_distance_list = {}

trained_model_using = ""
if "ped1" in args.pred_model_dir:
    trained_model_using = "ped1"
elif "ped2" in args.pred_model_dir:
    trained_model_using = "ped2"
elif "avenue" in args.pred_model_dir:
    trained_model_using = "avenue"

print('Start Evaluation of:', args.dataset_type + ',',
      'method: recon+pred,', 'trained model used:', trained_model_using)

# setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    # pred without first (t_length - 1) frames
    from_frame_pred = (args.t_length-1)+label_length
    to_frame_pred = videos[video_name]['length']+label_length
    frame_labels_pred = labels[0][from_frame_pred:to_frame_pred]
    labels_list_pred = np.append(labels_list_pred, frame_labels_pred)

    # pred with full frames
    from_frame = label_length
    to_frame = videos[video_name]['length']+label_length
    frame_labels_full = labels[0][from_frame:to_frame]
    labels_list_full = np.append(labels_list_full, frame_labels_full)

    # update indices
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    recon_psnr_list[video_name] = []
    feature_distance_list[video_name] = []
    recon_feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
pred_m_items_test = pred_m_items.clone()

pred_model.eval()

output_dir = os.path.join('./dataset', args.dataset_type, 'output')
output_frames_dir = os.path.join(output_dir, 'frames')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# Iterate on each frame of the whole dataset, forward through the model
index_image_output = 0
pre_label_length = 0
for k, (imgs) in enumerate(test_batch):
    if k == label_length - (args.t_length - 1) * (video_num + 1):
        video_num += 1
        pre_label_length = label_length - (args.t_length - 1) * (video_num)
        label_length += videos[videos_list[video_num]
                               .split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    if k == pre_label_length:
        for i in range(args.t_length):
            # do recon
            imgs_input = imgs[:, (3*i):(3*(i+1))]

            outputs, feas, updated_feas, recon_m_items, softmax_score_query, softmax_score_memory, compactness_loss = recon_model.forward(
                imgs_input, recon_m_items, False)
            mse_imgs = torch.mean(loss_func_mse(
                (outputs[0]+1)/2, (imgs_input[0]+1)/2)).item()
            mse_feas = compactness_loss.item()

            # Calculating the threshold for updating at the test time
            recon_point_sc = point_score(outputs, imgs_input)

            if recon_point_sc < args.recon_th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)  # b X h X w X d
                recon_m_items = recon_model.memory.update(
                    query, recon_m_items, False)

            # calculate psnr for each frame and then append it to psnr list
            psnr_score = psnr(mse_imgs)
            psnr_index = videos_list[video_num].split('/')[-1]
            recon_psnr_list[psnr_index].append(psnr_score)
            # append compactness lost of current frame to compactness list
            recon_feature_distance_list[videos_list[video_num].split(
                '/')[-1]].append(mse_feas)

            if args.is_save_output == 'true':
                num_frame = len(test_batch)
                num_digit_of_num_frame = len(str(num_frame))

                img_out_clone = torch.clone(outputs)
                img_out_clone = img_out_clone[0].permute(1, 2, 0)
                img_out_clone = img_out_clone.cpu().detach().numpy()

                img_out_clone = (img_out_clone + 1) * 127.5  # revert range
                img_out_clone = img_out_clone.astype(dtype=np.uint8)

                img_name_dir = ""
                if num_digit_of_num_frame == 3:
                    img_name_dir = output_frames_dir + "/%03d.jpg" % index_image_output
                elif num_digit_of_num_frame == 4:
                    img_name_dir = output_frames_dir + "/%04d.jpg" % index_image_output
                elif num_digit_of_num_frame == 5:
                    img_name_dir = output_frames_dir + "/%05d.jpg" % index_image_output
                else:
                    img_name_dir = output_frames_dir + "/%d.jpg" % index_image_output
                index_image_output += 1

                cv2.imwrite(img_name_dir, img_out_clone)
    else:
        # do recon
        imgs_input = imgs[:, 3*(args.t_length-1):]

        outputs, feas, updated_feas, recon_m_items, softmax_score_query, softmax_score_memory, compactness_loss = recon_model.forward(
            imgs_input, recon_m_items, False)
        mse_imgs = torch.mean(loss_func_mse(
            (outputs[0]+1)/2, (imgs_input[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        recon_point_sc = point_score(outputs, imgs_input)

        if recon_point_sc < args.recon_th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 1)  # b X h X w X d
            recon_m_items = recon_model.memory.update(
                query, recon_m_items, False)

        # calculate psnr for each frame and then append it to psnr list
        psnr_score = psnr(mse_imgs)
        psnr_index = videos_list[video_num].split('/')[-1]
        recon_psnr_list[psnr_index].append(psnr_score)
        # append compactness lost of current frame to compactness list
        recon_feature_distance_list[videos_list[video_num].split(
            '/')[-1]].append(mse_feas)

    outputs, feas, updated_feas, pred_m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = pred_model.forward(
        imgs[:, 0:3*(args.t_length-1)], pred_m_items_test, False)
    mse_imgs = torch.mean(loss_func_mse(
        (outputs[0]+1)/2, (imgs[0, 3*(args.t_length-1):]+1)/2)).item()
    mse_feas = compactness_loss.item()

    # Calculating the threshold for updating at the test time
    point_sc = point_score(outputs, imgs[:, 3*(args.t_length-1):])

    if args.is_save_output == 'true':
        num_frame = len(test_batch)
        num_digit_of_num_frame = len(str(num_frame))

        img_out_clone = torch.clone(outputs)
        img_out_clone = img_out_clone[0].permute(1, 2, 0)
        img_out_clone = img_out_clone.cpu().detach().numpy()

        img_out_clone = (img_out_clone + 1) * 127.5  # revert range
        img_out_clone = img_out_clone.astype(dtype=np.uint8)

        img_name_dir = ""
        if num_digit_of_num_frame == 3:
            img_name_dir = output_frames_dir + "/%03d.jpg" % index_image_output
        elif num_digit_of_num_frame == 4:
            img_name_dir = output_frames_dir + "/%04d.jpg" % index_image_output
        elif num_digit_of_num_frame == 5:
            img_name_dir = output_frames_dir + "/%05d.jpg" % index_image_output
        else:
            img_name_dir = output_frames_dir + "/%d.jpg" % index_image_output
        index_image_output += 1
        cv2.imwrite(img_name_dir, img_out_clone)

    if point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        pred_m_items_test = pred_model.memory.update(
            query, pred_m_items_test, False)

    # calculate psnr for each frame and then append it to psnr list
    psnr_score = psnr(mse_imgs)
    psnr_index = videos_list[video_num].split('/')[-1]
    psnr_list[psnr_index].append(psnr_score)
    # append compactness lost of current frame to compactness list
    feature_distance_list[videos_list[video_num].split(
        '/')[-1]].append(mse_feas)

    if k % 1000 == 0:
        print('DONE:', k, "frames")


# Measuring the abnormality score and the AUC for task: recon, pred without (t_length - 1) first frames
recon_anomaly_score_total_list_per_video = {}
anomaly_score_total_list_recon = []
anomaly_score_total_list_pred = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]

    # Score for recon
    recon_psnr_list_of_video = recon_psnr_list[video_name]
    # min-max normalization for PSNR
    recon_anomaly_score_list_of_video = anomaly_score_list(
        recon_psnr_list_of_video)

    recon_feature_distance_list_of_video = recon_feature_distance_list[video_name]
    # min-max normalization for compactness loss
    recon_anomaly_score_list_inv_of_video = anomaly_score_list_inv(
        recon_feature_distance_list_of_video)

    # Sum score for anomaly rate
    recon_score = score_sum(recon_anomaly_score_list_of_video,
                            recon_anomaly_score_list_inv_of_video, args.recon_alpha)

    # Score for pred
    psnr_list_of_video = psnr_list[video_name]
    # min-max normalization for PSNR
    anomaly_score_list_of_video = anomaly_score_list(psnr_list_of_video)

    feature_distance_list_of_video = feature_distance_list[video_name]
    # min-max normalization for compactness loss
    anomaly_score_list_inv_of_video = anomaly_score_list_inv(
        feature_distance_list_of_video)

    # Sum score for anomaly rate
    pred_score = score_sum(anomaly_score_list_of_video,
                           anomaly_score_list_inv_of_video, args.alpha)

    # Append score to total list
    index_last_frame = args.t_length - 1

    anomaly_score_total_list_recon += recon_score
    recon_anomaly_score_total_list_per_video[video_name] = recon_score

    anomaly_score_total_list_pred += pred_score

recon_accuracy = AUC(anomaly_score_total_list_recon,
                     np.expand_dims(1-labels_list_full, 0))
recon_opt_threshold = optimal_threshold(
    anomaly_score_total_list_recon, labels_list_full)
pred_opt_threshold = optimal_threshold(
    anomaly_score_total_list_pred, labels_list_pred)

pred_nomaly_average_score, pred_anomaly_average_score = average_score(
    anomaly_score_total_list_pred, pred_opt_threshold)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    # Score for pred
    psnr_list_of_video = psnr_list[video_name]
    # min-max normalization for PSNR
    anomaly_score_list_of_video = anomaly_score_list(psnr_list_of_video)

    feature_distance_list_of_video = feature_distance_list[video_name]
    # min-max normalization for compactness loss
    anomaly_score_list_inv_of_video = anomaly_score_list_inv(
        feature_distance_list_of_video)

    # Sum score for anomaly rate
    pred_score = score_sum(anomaly_score_list_of_video,
                           anomaly_score_list_inv_of_video, args.alpha)

    # Append score to total list

    for i in range(args.t_length - 1):
        if recon_anomaly_score_total_list_per_video[video_name][i] < recon_opt_threshold:
            anomaly_score_total_list += [pred_anomaly_average_score]
        else:
            anomaly_score_total_list += [pred_nomaly_average_score]

    anomaly_score_total_list += pred_score


anomaly_score_total_list = np.asarray(anomaly_score_total_list)

print('Number of frames:', len(labels[0]))
print('len of anomaly score:', len(anomaly_score_total_list))

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list_full, 0))

log_dir = os.path.join('./exp', args.dataset_type, "recon+pred", args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

plot_ROC(anomaly_score_total_list, np.expand_dims(
    1-labels_list_full, 0), accuracy, log_dir, args.dataset_type, "recon+pred", trained_model_using)

plot_anomaly_scores(anomaly_score_total_list,
                    labels[0], log_dir, args.dataset_type, "recon+pred", trained_model_using)

np.save(os.path.join(output_dir, 'recon_pred_anomaly_score.npy'),
        anomaly_score_total_list)

print('The result of', args.dataset_type)
print('AUC:', accuracy*100, '%')
print('recon_AUC:', recon_accuracy)
print('recon_opt_threshold:', recon_opt_threshold)
print('pred_opt_threshold:', pred_opt_threshold)
print('pred_nomaly_average_score:', pred_nomaly_average_score)
print('pred_anomaly_average_score:', pred_anomaly_average_score)


end_time = datetime.now()
time_range = end_time-start_time
print('Evaluate is taken: ', time_range)
