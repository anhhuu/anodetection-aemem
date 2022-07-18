from common import const
from video_capture import VideoCapture as vc
import argparse
import threading
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)

from matplotlib.animation import FuncAnimation
import tkinter as tk
import cv2
from image_similarity import ImageDifference as id
import numpy as np
from PIL import ImageTk, Image
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def mini_frame_coord(window_H, window_W, frame_h, frame_w):
    minus_h = window_H - frame_h
    minus_w = window_W - frame_w
    bias_h = minus_h/2
    bias_w = minus_w/2
    return bias_h, bias_w


class App:
    def __init__(self, window, window_title, dataset_type=const.DEFAULT_DATASET_NAME, t_length=const.DEFAULT_T_LENGTH):
        # create a window that contains everything on it
        self.window = window  # window = tk.Tk()
        self.window.iconbitmap()
        # set title for window
        self.window.wm_title("Anomaly Detection Application")
        self.dataset_type = dataset_type

        # create canvas to show widget
        self.set_up_canvas()

        self.current_data_path = ['./dataset/' + self.dataset_type + '/', ]

        # create a video source (by default this will try to open the computer webcam)
        self.vid = vc.VideoCapture(
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

    def static_update(self):
        # Get a frame from the video source
        test_frame, predicted_frame, anomaly_score = self.vid.get_static_frame_for_app(
            self.iter_frame)

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

        delay_time = const.DEFAULT_DELAY

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
        label = tk.Label(text="Anomaly Score Graph", font=const.LARGE_FONT)
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


# Create a window and pass it to the Application object
parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--dataset_type', type=str, default='ped2',
                    help='type of dataset: ped1, ped2, avenue')

args = parser.parse_args()
App(tk.Tk(), "Tkinter and OpenCV",
    dataset_type=args.dataset_type, t_length=args.t_length)
