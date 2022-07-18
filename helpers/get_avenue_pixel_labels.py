import numpy as np
import h5py
from scipy.io import loadmat
import cv2
import os

for videoidx in range(21):
    labelFolder = "./dataset/avenue/testing/pixel_masks/" + \
        "%02d.mat" % (videoidx + 1)
    data = loadmat(labelFolder)
    arr = np.zeros(0, np.int64)
    label_frames = data['volLabel'][0]
    num_digit_of_num_frame = len(str(len(label_frames)))

    output_dir = os.path.join('./dataset', 'avenue',
                              'testing', 'labels', '%02d' % (videoidx + 1))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(label_frames)):
        if num_digit_of_num_frame == 3:
            img_name_dir = output_dir + "/%03d.bmp" % i
        elif num_digit_of_num_frame == 4:
            img_name_dir = output_dir + "/%04d.bmp" % i
        elif num_digit_of_num_frame == 5:
            img_name_dir = output_dir + "/%05d.bmp" % i
        else:
            img_name_dir = output_dir + "/%d.bmp" % i
        cv2.imwrite(img_name_dir, label_frames[i]*255)

