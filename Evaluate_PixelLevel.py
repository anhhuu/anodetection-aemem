from re import X
from tkinter import ANCHOR
from typing import OrderedDict
import cv2
import os

from regex import D
from image_similarity import ImageDifference as id
import numpy as np
from sklearn import metrics
import imutils
from utils import *

# imageA = cv2.imread('./dataset/ped2/testing/frames/01/104.jpg')
# imageA = cv2.resize(imageA, (256, 256))
# imageB = cv2.imread('./dataset/ped2/output/frames/0100.jpg')
# imageB = cv2.resize(imageB, (256, 256))
# cv2.imshow("label frame", cv2.resize(imageA, (256, 256)))
# cv2.imshow("predict frame", cv2.resize(imageB, (256, 256)))


#test_frame = cv2.imread('./dataset/ped2/testing/frames/01/104.jpg')
#test_frame = cv2.resize(test_frame, (256, 256))
#pixel_label_frame = labels[104]
#predicted_frame = cv2.imread('./dataset/ped2/output/frames/0100.jpg')
#predicted_frame = cv2.resize(predicted_frame, (256, 256))

# Show images
#cv2.imshow("test", cv2.resize(test_frame, (256, 256)))
#cv2.imshow("predicted", cv2.resize(predicted_frame, (256, 256)))
#cv2.imshow("pixel_label", cv2.resize(pixel_label_frame, (256, 256)))


DEFAULT_DATASET_NAME = 'ped2'
DEFAULT_FRAME_PRED_INPUT = 4
DEFAULT_DATAPATH = ['./dataset/ped2/', ]
class VideoCapture:
    def __init__(self, data_path=DEFAULT_DATAPATH, dataset_type=DEFAULT_DATASET_NAME, frame_sequence_length=DEFAULT_FRAME_PRED_INPUT):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.frame_sequence_length = frame_sequence_length

        # load test and predicted frames
        test_frames, predicted_framese, test_num_video_index = self.get_dataset_frames()
        self.vid = [test_frames, predicted_framese, test_num_video_index]
        # load frame
        self.frame_scores = np.load(self.data_path[-1] + 'output/anomaly_score.npy')
        self.labels = np.load('./data/frame_labels_' + self.dataset_type + '.npy')
        self.pixelLabels = self.load_pixelLabel_frames()
        self.opt_threshold = self.optimalThreshold(self.frame_scores, self.labels)
        self.pixel_detected_frames = self.load_pixel_detected_frames()

    def get_static_frame(self, iter_frame):
        test_video_index = self.vid[2]
        
        # load the two input images
        i = iter_frame

        test_frame_list = self.vid[0] 

        # Get predicted frame
        pred_frame_list = self.vid[1] 
        current_pred_frame = pred_frame_list[i]

        # Get ground-truth frame
        map_index = test_video_index[i]
        true_index_of_test_frame = i + self.frame_sequence_length * test_video_index[i+map_index*4]
        current_test_frame = test_frame_list[true_index_of_test_frame]

        # Get pixel-level label frame
        pixel_detected_frame = self.pixel_detected_frames[i]
        pixel_level_label = self.pixelLabels[true_index_of_test_frame]

        # Get label score
        label_score = self.labels[0, true_index_of_test_frame]

        # resize image
        w1, h1, c1 = current_test_frame.shape
        w2, h2, c2 = current_pred_frame.shape
        if w1 != 256:
            current_test_frame = cv2.resize(current_test_frame, (256, 256))
        if w2 != 256:
            current_pred_frame = cv2.resize(current_pred_frame, (256, 256))

        anomaly_score = self.frame_scores[i]
        return current_test_frame, current_pred_frame, anomaly_score, pixel_level_label, pixel_detected_frame, label_score

    def get_dataset_frames(self):
        time_t = 0
        test_input_path = []
        test_video_dir = []
        test_video_dir_distinct = []
        for path, _, files in os.walk(self.data_path[-1] + 'testing/frames'):
            for name in files:
                if(path not in test_video_dir_distinct):
                    test_video_dir_distinct.append(path)
                test_input_path.append(os.path.join(path, name))
                test_video_dir.append(path)
        test_input_path.sort()
        test_video_dir.sort()
        test_video_dir_distinct.sort()

        test_video_dir_distinct_map_index = {}
        for i in range(len(test_video_dir_distinct)):
            test_video_dir_distinct_map_index[test_video_dir_distinct[i]] = i + 1

        for i in range(len(test_video_dir)):
            test_video_dir[i] = test_video_dir_distinct_map_index[test_video_dir[i]]

        pred_input_path = []
        for path, _, files in os.walk(self.data_path[-1] + 'output/frames'):
            for name in files:
                pred_input_path.append(os.path.join(path, name))
        pred_input_path.sort()

        test_input_imgs = []
        for i in range(len(test_input_path)):
            img = cv2.imread(test_input_path[i])
            test_input_imgs.append(img)

        pred_input_imgs = []
        for i in range(len(pred_input_path)):
            img = cv2.imread(pred_input_path[i])
            pred_input_imgs.append(img)

        return test_input_imgs, pred_input_imgs, test_video_dir

    def optimalThreshold(self, anomal_scores, labels):
        y_true = 1 - labels[0, :1962]
        y_true = np.squeeze(y_true)
        y_score = np.squeeze(anomal_scores[:1962])
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
        frame_auc = metrics.roc_auc_score(y_true, y_score)
        print("AUC: {}".format(frame_auc))
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
        return threshold[ix]

    def load_pixelLabel_frames(self):
        label_input_path = []
        label_dir = []
        label_dir_distinct = []
        cur_path = './dataset/' + self.dataset_type + '/testing/labels'
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
            label_img.astype("uint8")
            label_img = cv2.resize(label_img, (256, 256))
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
            label_list.append(label_img)

        return label_list
    
    def load_pixel_detected_frames(self):
        input_path = []
        directory = []
        dir_distinct = []
        current_path = './dataset/' + self.dataset_type + '/output/detected_regions'
        for path, _, files in os.walk(current_path):
            for name in files:
                if(path not in dir_distinct):
                    dir_distinct.append(path)
                input_path.append(os.path.join(path, name))
                directory.append(path)
        input_path.sort()
        directory.sort()
        dir_distinct.sort()

        pixel_detected_frames = []
        for i in range(len(input_path)):
            detected_img = cv2.imread(input_path[i])
            detected_img.astype("uint8")
            detected_img = cv2.resize(detected_img, (256, 256))
            detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)
            pixel_detected_frames.append(detected_img)

        return pixel_detected_frames

    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def get_boundingbox(self, binary_image):
        detected_bbox = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_bbox = imutils.grab_contours(detected_bbox)
        return detected_bbox

    def show_bbox(self, detected_bbox):
        show_image = np.zeros((256, 256), np.uint8)
        for c in detected_bbox:
            (x1, y1, w1, h1) = cv2.boundingRect(c)
            # Remove regions in-significant regions
            if w1 > 17 or h1 > 17: 
                cv2.rectangle(regions_detected, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 1)
                for ii in range(w1):
                    real_x = x1 + ii
                    for j in range(h1):     
                        real_y = y1 + j
                        show_image[(real_y, real_x)] = 255

        return show_image

def create_detected_pixelLevel_frame(test, predicted, anomaly_score, optimal_threshold):
    ID = id.ImageDifference()
    # *** PIXEL LEVEL
    test_img_detected, pred_img_detected, complete_process_img, SSIM_diff_img, SSIM_score = ID.image_differences(
    test_frame, predicted_frame, anomaly_score, DATA.opt_threshold)

    cv2.imshow("pixel_label", cv2.resize(complete_process_img, (256, 256)))
    #ID.compare_image_diff(pixel_label_frame, thresholded_img)


DATA = VideoCapture()
ID = id.ImageDifference()
optimal_threshold = DATA.opt_threshold
output_frames_dir = './dataset/' + DATA.dataset_type + '/output/detected_regions'
save_pixelLevel_detected_frames = False
measure_pixelLevel_AUC = True

if save_pixelLevel_detected_frames == True:
    for i in range(len(DATA.vid[1])):
        # Get a frame from the video source
        print("Frame th: ", i)
        test_frame, predicted_frame, anomaly_score, pixel_label_frame = DATA.get_static_frame(i)
        create_detected_pixelLevel_frame(test_frame, predicted_frame, anomaly_score, optimal_threshold)

        # *** PIXEL LEVEL
        test_img_detected, pred_img_detected, complete_process_img, SSIM_diff_img, SSIM_score = ID.image_differences(
        test_frame, predicted_frame, anomaly_score, DATA.opt_threshold)

        #if i == 144:
            #cv2.imshow('image_differences', complete_process_img)
        #ID.compare_image_diff(pixel_label_frame, thresholded_img)
        #self.write_score.save_score(SSIM_score)

        num_frame = 2010
        num_digit_of_num_frame = len(str(num_frame))

        img_name_dir = ""
        img_name_dir = output_frames_dir + "/%04d.jpg" % i
        # if num_digit_of_num_frame == 3:
        #     img_name_dir = output_frames_dir + "/%03d.jpg" % i
        # elif num_digit_of_num_frame == 4:
        #     img_name_dir = output_frames_dir + "/%04d.jpg" % i
        # elif num_digit_of_num_frame == 5:
        #     img_name_dir = output_frames_dir + "/%05d.jpg" % i
        # else:
        #     img_name_dir = output_frames_dir + "/%d.jpg" % i

        cv2.imwrite(img_name_dir, complete_process_img)

IoU_score_list = []
if measure_pixelLevel_AUC == True:
    labels_list = []
    for i in range(len(DATA.vid[1])):
        test_frame, predicted_frame, anomaly_score, pixel_level_label, pixel_detected_frame, label_score = DATA.get_static_frame(i)
        labels_list.append(label_score)
        if anomaly_score > (optimal_threshold):
            IoU_score_list.append(0)
            continue
        else:
            label_bbox_raw = DATA.get_boundingbox(pixel_level_label)
            detected_bbox_raw = DATA.get_boundingbox(pixel_detected_frame)
            # loop over the contours in order to show bbox as well as eliminate small regions 

            #cv2.imshow("detected_frame", pixel_detected_frame)
            regions_detected = pixel_detected_frame.copy()
    
            detected_bbox = DATA.show_bbox(detected_bbox_raw)
            label_bbox =  DATA.show_bbox(label_bbox_raw)

            # cv2.waitKey(0)
            Intersection_Area = np.logical_and(label_bbox, detected_bbox)
            Union_Area = np.logical_or(label_bbox, detected_bbox)
            Inter_sum = np.sum(Intersection_Area)
            Uni_sum = np.sum(Union_Area)
            IoU_score = 0
            if Uni_sum == 0:
               IoU_score = 0
            else:
               IoU_score =  Inter_sum / Uni_sum
            IoU_score_list.append(IoU_score)

            # if i == 59 or i == 61 or i == 70:
            #    cv2.imshow("detect_bbox", detected_bbox)
            #    cv2.imshow("detected", pixel_detected_frame)
            #    cv2.imshow("regions_detected", regions_detected)
            #    cv2.imshow("label", pixel_level_label)
            #    cv2.imshow("label_bbox", label_bbox)
            #    print("IoU: ", IoU_score)
            #    cv2.waitKey(0)
            #print("IOU = ", IoU_score)
            #cv2.waitKey(0)


    #label_array = DATA.labels[0, :1962]
    label_array = np.array(labels_list)
    count_tp = 0
    count_tn = 0
    for i in range(1962):
        if IoU_score_list[i] >= 0.4 and labels_list[i] == 1:
            count_tp += 1
        elif IoU_score_list[i] <          0.4 and labels_list[i] == 0:
            count_tn += 1
    
    accuracy = (count_tp + count_tn)/1962
    print('accuracy:', accuracy*100, '%')
    #AUC = AUC(IoU_score_list, np.expand_dims(1-label_array, 0))
    #print('AUC:', AUC*100, '%')
    