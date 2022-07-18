from cProfile import label
import cv2
import os
from image_similarity import ImageDifference as id
from video_capture import VideoCapture as vc
import numpy as np
from utils import *
from common import const


save_pixel_level_detected_frames = True
measure_pixel_level_AUC = not save_pixel_level_detected_frames


def create_detected_pixelLevel_frame(test, predicted, anomaly_score, optimal_threshold):
    ID = id.ImageDifference()
    _, _, complete_process_img, _, _ = ID.image_differences(
        test_frame, predicted_frame, anomaly_score, DATA.opt_threshold)

    cv2.imshow("pixel_label", cv2.resize(complete_process_img, (256, 256)))
    # ID.compare_image_diff(pixel_label_frame, thresholded_img)


DATA = vc.VideoCapture()
ID = id.ImageDifference()
optimal_threshold = DATA.opt_threshold[0]
output_frames_dir = './dataset/' + DATA.dataset_type + '/output/detected_regions'

is_exist = os.path.exists(output_frames_dir)

if not is_exist:
    os.makedirs(output_frames_dir)

if save_pixel_level_detected_frames == True:
    for i in range(len(DATA.vid[1])):
        # Get a frame from the video source
        print("Frame th: ", i)
        test_frame, predicted_frame, anomaly_score, pixel_label_frame = DATA.get_static_frame_for_export_label(
            i)
        create_detected_pixelLevel_frame(
            test_frame, predicted_frame, anomaly_score, optimal_threshold)

        # PIXEL LEVEL
        test_img_detected, pred_img_detected, complete_process_img, SSIM_diff_img, SSIM_score = ID.image_differences(
            test_frame, predicted_frame, anomaly_score, DATA.opt_threshold)

        img_name_dir = ""
        img_name_dir = output_frames_dir + "/%04d.jpg" % i

        cv2.imwrite(img_name_dir, complete_process_img)

IoU_score_list = []
if measure_pixel_level_AUC == True:
    labels_list = []
    for i in range(len(DATA.vid[1])):
        test_frame, predicted_frame, anomaly_score, pixel_level_label, pixel_detected_frame, label_score = DATA.get_static_frame_for_evaluate(
            i)
        labels_list.append(label_score)
        if anomaly_score > (optimal_threshold):
            IoU_score_list.append(0)
            continue
        else:
            label_bbox_raw = DATA.get_bounding_box(pixel_level_label)
            detected_bbox_raw = DATA.get_bounding_box(pixel_detected_frame)
            # Loop over the contours in order to show bbox as well as eliminate small regions

            regions_detected = pixel_detected_frame.copy()

            detected_bbox = DATA.show_bbox(detected_bbox_raw, regions_detected)
            label_bbox = DATA.show_bbox(label_bbox_raw, regions_detected)

            intersection_area = np.logical_and(label_bbox, detected_bbox)
            label_area = np.logical_and(label_bbox, label_bbox)
            label_sum = np.sum(label_area)
            inter_sum = np.sum(intersection_area)
            IoU_score = 0
            if label_sum == 0:
                IoU_score = 0
            else:
                IoU_score = inter_sum / label_sum
            IoU_score_list.append(IoU_score)

    label_array = np.array(labels_list)
    count_tp = 0
    count_tn = 0

    len_labels = 0
    if const.DEFAULT_DATASET_NAME == "avenue":
        len_labels = 15240
    if const.DEFAULT_DATASET_NAME == "ped2":
        len_labels = 1962

    for i in range(len_labels):
        if IoU_score_list[i] >= 0.4:  # and labels_list[i] == 1:
            count_tp += 1
        elif IoU_score_list[i] < 0.4 and labels_list[i] == 0:
            count_tn += 1

    accuracy = (count_tp + count_tn)/len_labels
    print('accuracy:', accuracy*100, '%')
