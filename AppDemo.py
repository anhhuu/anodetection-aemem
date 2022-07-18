import argparse
import threading
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

from matplotlib.animation import FuncAnimation
import tkinter as tk
import os
import cv2
from image_similarity import ImageDifference as id
import numpy as np
from PIL import ImageTk, Image
import time
import matplotlib.pyplot as plt
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)

DEFAULT_DATASET_NAME = "ped2"
DEFAULT_METHOD = "pred"
DEFAULT_FRAME_PRED_INPUT = 4
DEFAULT_T_LENGTH = 5
DEFAULT_DELAY = 15


def mini_frame_coord(window_H, window_W, frame_h, frame_w):
    minus_h = window_H - frame_h
    minus_w = window_W - frame_w
    bias_h = minus_h/2
    bias_w = minus_w/2
    return bias_h, bias_w


class App:
    def __init__(self, window, window_title, dataset_type=DEFAULT_DATASET_NAME, method=DEFAULT_METHOD, t_length=DEFAULT_T_LENGTH):
        # create a window that contains everything on it
        self.window = window  # window = tk.Tk()
        self.window.iconbitmap()
        # set title for window
        self.window.wm_title("Anomaly Detection Application")
        self.dataset_type = dataset_type
        self.method = method

        # create canvas to show widget
        self.set_up_canvas()

        # create button for file loading process
        self.setup_buttons()

        # create a video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(
            self.current_data_path, self.dataset_type,  frame_sequence_length=t_length-1)
        self.numPredFrame = len(self.vid.vid[1])

        # create an ImgDiff object for do difference on images
        self.ImgDiff = id.ImageDifference()

        self.static_update()
        threading.Thread(target=self.static_update_figure()).start()

        self.window.protocol('WM_DELETE_WINDOW', self.quit)
        self.window.mainloop()

    def set_up_canvas(self):
        # Create a canvas that can fit the above video source size
        self.canvas_H, self.canvas_W = 350, 1200
        self.canvas_center_H, self.canvas_center_W = self.canvas_H/2, self.canvas_W/2
        self.canvas = tk.Canvas(
            self.window, width=self.canvas_W, height=self.canvas_H, background="#4E747E")
        self.canvas.pack()

        # Setup for timer
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.iter_frame = 0
        self.prev_frame_time = 0  # used to record the time when we processed last frame
        self.new_frame_time = 0  # used to record the time at which we processed current frame

        # Set up for showing frames
        # Bias height and bias width for add mini frames canvas
        self.bias_h, self.bias_w = mini_frame_coord(
            self.canvas_H, self.canvas_W, 256, 256)
        self.frame_1_x_axis = 35
        self.frame_2_x_axis = 35+256+35
        self.frame_3_x_axis = 35+256+35+256+35
        self.frame_4_x_axis = 35+256+35+256+35+256+35
        self.fps = self.canvas.create_text(
            450, 340, fill="white", font="Times 15 italic bold", text="fps: ")
        self.frame_th = self.canvas.create_text(
            550, 340, fill="white", font="Times 15 italic bold", text="frame: ")
        self.anomaly_tag = self.canvas.create_text(
            750, 340, fill="white", font="Times 30 italic bold", text="Abnormal: ")
        self.dataset_name_tag = self.canvas.create_text(
            35, 320, fill="white", font='Helvatica 20 bold', text="Dataset: Ped2", anchor=tk.NW)

        # Show name of each kind of showing frame
        self.canvas.create_text(self.frame_1_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Ground-truth", anchor=tk.NW)
        self.canvas.create_text(self.frame_2_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Recons-Error", anchor=tk.NW)
        self.canvas.create_text(self.frame_3_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Thresholded", anchor=tk.NW)
        if self.dataset_type == 'ped2':
            self.canvas.create_text(self.frame_4_x_axis, 10, fill="white",
                                    font='Helvatica 20 bold', text="Anomaly Regions", anchor=tk.NW)
        else:
            self.canvas.create_text(self.frame_4_x_axis, 10, fill="white",
                                    font='Helvatica 20 bold', text="Anomaly Detection", anchor=tk.NW)

    def change_dataset(self):
        folder_path = filedialog.askdirectory()
        self.current_data_path.append(folder_path)
        current_dataset_name = folder_path.split('/')[-1]
        self.canvas.itemconfig(self.dataset_name_tag, fill='white',
                               text="Dataset: {}".format(current_dataset_name))

    def setup_buttons(self):
        self.current_data_path = ['./dataset/' + self.dataset_type + '/', ]
        # Button that lets the user take a snapshot
        self.open_dataset = tk.Button(self.window, text="Open Dataset", width=50, command=self.change_dataset,
                                      fg='white', bg="#263D42")
        self.open_dataset.pack(anchor=tk.CENTER, expand=True)

    def static_update(self):
        # Get a frame from the video source
        test_frame, predicted_frame, anomaly_score = self.vid.get_static_frame(
            self.iter_frame)
        #test_frame, predicted_frame, anomaly_score, pixel_label_frame = self.vid.get_static_frame(self.iter_frame)

        optimal_threshold = self.vid.opt_threshold
        # Calculate difference image

        test_img_detected, _, thresholded_img, SSIM_diff_img, _ = self.ImgDiff.image_differences(
            test_frame, predicted_frame, anomaly_score, self.vid.opt_threshold)

        # Closes all the frames time when we finish processing for this frame
        self.new_frame_time = time.time()
        # Calculate frame per seccond value
        fps = 1/(self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

        # Show information on canvas
        self.canvas.itemconfig(self.fps, text="fps: {}".format(
            int(fps)))  # Update fps on canvas
        self.canvas.itemconfig(
            self.frame_th, text="frame: {}".format(self.iter_frame))
        if anomaly_score < optimal_threshold:
            self.canvas.itemconfig(
                self.anomaly_tag, fill='red', text="Abnormal: YES")
        else:
            self.canvas.itemconfig(
                self.anomaly_tag, fill='white', text="Abnormal: NO")

        delay_time = DEFAULT_DELAY

        # SHOW BOUNDING BOX ON PED2 -- SHOW ANOMALY DETECT RESULT ON REMAINS
        if self.dataset_type == 'ped1':
            delay_time = 50

            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255

            test_clone = test_frame.copy()
            h = test_clone.shape[0]
            w = test_clone.shape[1]
            if anomaly_score < self.vid.opt_threshold:
                cv2.rectangle(test_clone, (0, 0), (w, h), (255, 0, 0), 5)

            # Convert opencv narray images to PIL images
            self.photo_test = ImageTk.PhotoImage(
                image=Image.fromarray(test_frame))
            self.photo_pred = ImageTk.PhotoImage(
                image=Image.fromarray(SSIM_diff_img))
            self.photo_diff = ImageTk.PhotoImage(
                image=Image.fromarray(thresholded_img))
            self.detected_regions = ImageTk.PhotoImage(
                image=Image.fromarray(test_clone))

        elif self.dataset_type == 'ped2':
            delay_time = 20

            # Convert opencv narray images to PIL images
            self.photo_test = ImageTk.PhotoImage(
                image=Image.fromarray(test_frame))
            self.photo_pred = ImageTk.PhotoImage(
                image=Image.fromarray(SSIM_diff_img))
            self.photo_diff = ImageTk.PhotoImage(
                image=Image.fromarray(thresholded_img))
            self.detected_regions = ImageTk.PhotoImage(
                image=Image.fromarray(test_img_detected))

        elif self.dataset_type == 'avenue':
            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255

            test_clone = test_frame.copy()
            h = test_clone.shape[0]
            w = test_clone.shape[1]
            if anomaly_score < self.vid.opt_threshold:
                cv2.rectangle(test_clone, (0, 0), (w, h), (255, 0, 0), 5)

            # Convert opencv narray images to PIL images
            self.photo_test = ImageTk.PhotoImage(
                image=Image.fromarray(test_frame))
            self.photo_pred = ImageTk.PhotoImage(
                image=Image.fromarray(SSIM_diff_img))
            self.photo_diff = ImageTk.PhotoImage(
                image=Image.fromarray(thresholded_img))
            self.detected_regions = ImageTk.PhotoImage(
                image=Image.fromarray(test_clone))

        # Attach test, predicted, difference and detected_regions images on canvas
        self.canvas.create_image(
            self.frame_1_x_axis, self.bias_h, image=self.photo_test, anchor=tk.NW)
        self.canvas.create_image(
            self.frame_2_x_axis, self.bias_h, image=self.photo_pred, anchor=tk.NW)
        self.canvas.create_image(
            self.frame_3_x_axis, self.bias_h, image=self.photo_diff, anchor=tk.NW)
        self.canvas.create_image(
            self.frame_4_x_axis, self.bias_h, image=self.detected_regions, anchor=tk.NW)

        # Function callback
        self.window.after(delay_time, self.static_update)
        if self.iter_frame == len(self.vid.vid[1]):
            self.iter_frame = 0
        else:
            self.iter_frame += 1

    def static_animate(self, i):
        y_score = np.squeeze(self.vid.frame_scores[:self.iter_frame+1])

        len = y_score.size
        x = np.arange(0, len)

        # Declare a clear axis each time
        plt.cla()
        y_thresh = self.vid.opt_threshold
        # create a legend

        x_thresh = (0, self.numPredFrame)
        y_thresh = (y_thresh, y_thresh)
        plt.plot(x, y_score, color="steelblue", label='score/frame')
        plt.plot(x_thresh, y_thresh, color="red",
                 marker='o', label="threshold")
        plt.plot(len, y_score[len-1], color="green",
                 marker='o', label="current_frame")
        plt.legend(loc='lower left')
        plt.ylabel('Score')
        plt.xlabel('Frames')
        plt.tight_layout()

    def static_update_figure(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        # Set label for the figure
        label = tk.Label(text="Anomaly Score Graph", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        self.ani = FuncAnimation(
            self.figure, self.static_animate, interval=1000)
        # Create canvas that hold figure
        self.canvas_fig = FigureCanvasTkAgg(plt.gcf(), self.window)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def quit(self):
        self.canvas.destroy()
        for after_id in self.window.tk.eval('after info').split():
            self.window.after_cancel(after_id)
        self.window.destroy()
        cv2.destroyAllWindows()
        plt.close()


class VideoCapture:
    def __init__(self, data_path=[], dataset_type=DEFAULT_DATASET_NAME, frame_sequence_length=DEFAULT_FRAME_PRED_INPUT):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.frame_sequence_length = frame_sequence_length
        # Open the video source, # capture video by webcam by default

        # load test and predicted frames
        test_frames, predicted_framese, test_num_video_index = self.get_dataset_frames()
        self.vid = [test_frames, predicted_framese, test_num_video_index]

        # FRAME-LEVEL data
        self.frame_scores = np.load(
            self.data_path[-1] + 'output/anomaly_score.npy')
        self.labels = np.load(
            './data_labels/frame_labels_' + self.dataset_type + '.npy')
        self.opt_threshold = np.load(
            self.data_path[-1] + 'output/optimal_threshold.npy')

        # PIXEL-LEVEL data
        self.pixelLabels = self.load_pixelLabel_frames()
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


# Create a window and pass it to the Application object
parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--method', type=str, default='pred',
                    help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--dataset_type', type=str, default='avenue',
                    help='type of dataset: ped1, ped2, avenue, shanghai')

args = parser.parse_args()
App(tk.Tk(), "Tkinter and OpenCV", dataset_type=args.dataset_type,
    method=args.method, t_length=args.t_length)
