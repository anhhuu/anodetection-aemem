import cv2
import os
import numpy as np
import imutils
from utils import *

from common import const


class VideoCapture:
    def __init__(self, data_path=const.DEFAULT_DATASET_PATH, dataset_type=const.DEFAULT_DATASET_NAME, frame_sequence_length=const.DEFAULT_FRAME_PRED_INPUT):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.frame_sequence_length = frame_sequence_length

        # load test and predicted frames
        test_frames, predicted_framese, test_num_video_index = self.get_dataset_frames()
        self.vid = [test_frames, predicted_framese, test_num_video_index]
        # load frame
        self.frame_scores = np.load(
            self.data_path[-1] + 'output/anomaly_score.npy')
        self.labels = np.load('./data_labels/frame_labels_' +
                              self.dataset_type + '.npy')
        self.pixelLabels = self.load_pixel_label_frames()
        self.opt_threshold = np.load(
            self.data_path[-1] + 'output/optimal_threshold.npy')
        self.pixel_detected_frames = self.load_pixel_detected_frames()

    def get_static_frame_for_app(self, iter_frame):
        test_video_index = self.vid[2]

        # load the two input images
        i = iter_frame

        test_frame_list = self.vid[0]
        # Get predicted frame
        pred_frame_list = self.vid[1]
        current_pred_frame = pred_frame_list[i]

        # Get ground-truth frame
        map_index = test_video_index[i]
        true_index_of_test_frame = i + self.frame_sequence_length * \
            test_video_index[i+map_index*4]
        current_test_frame = test_frame_list[true_index_of_test_frame]

        # resize image
        w1, _, _ = current_test_frame.shape
        w2, _, _ = current_pred_frame.shape
        if w1 != 256:
            current_test_frame = cv2.resize(current_test_frame, (256, 256))
        if w2 != 256:
            current_pred_frame = cv2.resize(current_pred_frame, (256, 256))

        anomaly_score = self.frame_scores[i]
        # , pixel_level_label#, pixel_detected_frame
        return current_test_frame, current_pred_frame, anomaly_score

    def get_static_frame_for_evaluate(self, iter_frame):
        test_video_index = self.vid[2]

        # load the two input images
        i = iter_frame

        test_frame_list = self.vid[0]

        # Get predicted frame
        pred_frame_list = self.vid[1]
        current_pred_frame = pred_frame_list[i]

        # Get ground-truth frame
        map_index = test_video_index[i]
        true_index_of_test_frame = i + self.frame_sequence_length * \
            test_video_index[i+map_index*4]
        current_test_frame = test_frame_list[true_index_of_test_frame]

        # Get pixel-level label frame
        pixel_detected_frame = self.pixel_detected_frames[i]
        pixel_level_label = self.pixelLabels[true_index_of_test_frame]

        # Get label score
        label_score = self.labels[0, true_index_of_test_frame]

        # resize image
        w1, _, _ = current_test_frame.shape
        w2, _, _ = current_pred_frame.shape
        if w1 != 256:
            current_test_frame = cv2.resize(current_test_frame, (256, 256))
        if w2 != 256:
            current_pred_frame = cv2.resize(current_pred_frame, (256, 256))

        anomaly_score = self.frame_scores[i]

        return current_test_frame, current_pred_frame, anomaly_score, pixel_level_label, pixel_detected_frame, label_score

    def get_static_frame_for_export_label(self, iter_frame):
        test_video_index = self.vid[2]

        # load the two input images
        i = iter_frame

        test_frame_list = self.vid[0]

        # Get predicted frame
        pred_frame_list = self.vid[1]
        current_pred_frame = pred_frame_list[i]

        # Get ground-truth frame
        map_index = test_video_index[i]
        true_index_of_test_frame = i + self.frame_sequence_length * \
            test_video_index[i+map_index*4]
        current_test_frame = test_frame_list[true_index_of_test_frame]

        # Get pixel-level label frame

        pixel_level_label = self.pixelLabels[true_index_of_test_frame]

        # resize image
        w1, _, _ = current_test_frame.shape
        w2, _, _ = current_pred_frame.shape
        if w1 != 256:
            current_test_frame = cv2.resize(current_test_frame, (256, 256))
        if w2 != 256:
            current_pred_frame = cv2.resize(current_pred_frame, (256, 256))

        anomaly_score = self.frame_scores[i]
        return current_test_frame, current_pred_frame, anomaly_score, pixel_level_label

    def get_dataset_frames(self):
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

    def load_pixel_label_frames(self):
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

    def get_bounding_box(self, binary_image):
        detected_bbox = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_bbox = imutils.grab_contours(detected_bbox)
        return detected_bbox

    def show_bbox(self, detected_bbox, regions_detected):
        show_image = np.zeros((256, 256), np.uint8)
        for c in detected_bbox:
            (x1, y1, w1, h1) = cv2.boundingRect(c)
            # Remove regions in-significant regions
            if w1 > 17 or h1 > 17:
                cv2.rectangle(regions_detected, (x1, y1),
                              (x1 + w1, y1 + h1), (255, 0, 0), 1)
                for ii in range(w1):
                    real_x = x1 + ii
                    for j in range(h1):
                        real_y = y1 + j
                        show_image[(real_y, real_x)] = 255

        return show_image
