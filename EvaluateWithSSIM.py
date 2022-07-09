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
import torchgeometry as tgm

parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--test_batch_size', type=int,
                    default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=128,
                    help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred',
                    help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--alpha', type=float, default=0.6,
                    help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01,
                    help='threshold for test updating')
parser.add_argument('--num_workers_test', type=int, default=1,
                    help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2',
                    help='type of dataset: ped1, ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str,
                    default='./dataset', help='directory of data')
parser.add_argument('--model_dir', type=str,
                    default='./pre_trained_model/defaults/ped2_prediction_model.pth', help='directory of model')
parser.add_argument('--m_items_dir', type=str,
                    default='./pre_trained_model/defaults/ped2_prediction_keys.pt', help='directory of model')
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
loss_func_ssim = tgm.losses.SSIM(5, reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)
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
labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

trained_model_using = ""
if "ped1" in args.model_dir:
    trained_model_using = "ped1"
elif "ped2" in args.model_dir:
    trained_model_using = "ped2"
elif "avenue" in args.model_dir:
    trained_model_using = "avenue"
elif "shanghai" in args.model_dir:
    trained_model_using = "shanghai"

print('Start Evaluation of:', args.dataset_type + ',', 'method:',
      args.method + ',', 'trained model used:', trained_model_using)

# setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    if args.method == 'pred':
        from_frame = (args.t_length-1)+label_length
        to_frame = videos[video_name]['length']+label_length
        frame_labels = labels[0][from_frame:to_frame]
        labels_list = np.append(labels_list, frame_labels)
    else:
        from_frame = label_length
        to_frame = videos[video_name]['length']+label_length
        frame_labels = labels[0][from_frame:to_frame]
        labels_list = np.append(labels_list, frame_labels)

    # update indices
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

output_dir = os.path.join('./dataset', args.dataset_type, 'output')
output_frames_dir = os.path.join(output_dir, 'frames')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# Iterate on each frame of the whole dataset, forward through the model
# predict: img ndim = 4, shape ([1, 15, 256, 256])
# recons: img ndim = 4, shape ([1, 3, 256, 256])
for k, (imgs) in enumerate(test_batch):
    if args.method == 'pred':
        if k == label_length-(args.t_length-1)*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num]
                                   .split('/')[-1]]['length']
    else:
        if k == label_length:
            video_num += 1
            label_length += videos[videos_list[video_num]
                                   .split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    if args.method == 'pred':
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(
            imgs[:, 0:3*(args.t_length-1)], m_items_test, False)

        pred_image = (outputs[0]+1)/2
        pred_image = torch.unsqueeze(pred_image, 0)
        ground_truth_image = (imgs[0, 3*(args.t_length-1):]+1)/2
        ground_truth_image = torch.unsqueeze(ground_truth_image, 0)
        loss_SSIM = loss_func_ssim(ground_truth_image, pred_image)
        mse_imgs = torch.mean(loss_SSIM).item()
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
                img_name_dir = output_frames_dir + "/%03d.jpg" % k
            elif num_digit_of_num_frame == 4:
                img_name_dir = output_frames_dir + "/%04d.jpg" % k
            elif num_digit_of_num_frame == 5:
                img_name_dir = output_frames_dir + "/%05d.jpg" % k
            else:
                img_name_dir = output_frames_dir + "/%d.jpg" % k

            cv2.imwrite(img_name_dir, img_out_clone)

    else:
        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(
            imgs, m_items_test, False)
        mse_imgs = torch.mean(loss_func_mse(
            (outputs[0]+1)/2, (imgs[0]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

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
                img_name_dir = output_frames_dir + "/%03d.jpg" % k
            elif num_digit_of_num_frame == 4:
                img_name_dir = output_frames_dir + "/%04d.jpg" % k
            elif num_digit_of_num_frame == 5:
                img_name_dir = output_frames_dir + "/%05d.jpg" % k
            else:
                img_name_dir = output_frames_dir + "/%d.jpg" % k

            cv2.imwrite(img_name_dir, img_out_clone)

    if point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    # calculate psnr for each frame and then append it to psnr list
    psnr_score = psnr(mse_imgs)
    psnr_index = videos_list[video_num].split('/')[-1]
    psnr_list[psnr_index].append(psnr_score)
    # append compactness lost of current frame to compactness list
    feature_distance_list[videos_list[video_num].split(
        '/')[-1]].append(mse_feas)

    if k % 1000 == 0:
        print('DONE:', k, "frames")


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]

    psnr_list_of_video = psnr_list[video_name]
    # min-max normalization for PSNR
    anomaly_score_list_of_video = anomaly_score_list(psnr_list_of_video)

    feature_distance_list_of_video = feature_distance_list[video_name]
    # min-max normalization for compactness loss
    anomaly_score_list_inv_of_video = anomaly_score_list_inv(
        feature_distance_list_of_video)

    # Sum score for anomaly rate
    score = score_sum(anomaly_score_list_of_video,
                      anomaly_score_list_inv_of_video, args.alpha)

    # Append score to total list
    anomaly_score_total_list += score


anomaly_score_total_list = np.asarray(anomaly_score_total_list)
print('Number of frames:', len(labels[0]))
print('len of anomaly score:', len(anomaly_score_total_list))

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

plot_ROC(anomaly_score_total_list, np.expand_dims(
    1-labels_list, 0), accuracy, log_dir, args.dataset_type, args.method, trained_model_using)

plot_anomaly_scores(anomaly_score_total_list,
                    labels[0], log_dir, args.dataset_type, args.method, trained_model_using)

np.save(os.path.join(output_dir, 'anomaly_score.npy'), anomaly_score_total_list)

print('The result of', args.dataset_type)
print('AUC:', accuracy*100, '%')


end_time = datetime.now()
time_range = end_time-start_time
print('Evaluate is taken: ', time_range)
