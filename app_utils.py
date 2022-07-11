import cv2
import os
import csv
import pandas as pd
from utils import *

class write_score():
    def __init__(self) -> None:
        pass

    def __create_header__(self) -> None:
        self.fieldnames = ["SSIM_score"]
        with open('data.csv', 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            csv_writer.writeheader()

    def save_score(self, score=0.0):
        self.fieldnames=""
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            info = {"SSIM_score": score,}
            csv_writer.writerow(info)

    def load_score(self, filename):
        data= pd.read_csv(filename)
        headers = data.columns
        field_name = headers[0]
        data_source = data.SSIM_score
        values = data_source.values
        return values

if __name__ == "__main__":
    Ob = write_score()
    SSIM_values = Ob.load_score("data.csv")
    SSIM_values_expand = np.expand_dims(SSIM_values, 0)
    labels = np.load('./data/frame_labels_ped2' + '.npy')
    label_cut = labels[0][:1962]
    label_expand = np.expand_dims(1-label_cut, 0)

    accuracy = AUC(SSIM_values_expand, label_expand)

    log_dir = os.path.join('./exp', 'ped2', 'pred', 'ped2')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    plot_ROC(SSIM_values_expand, label_expand, accuracy, log_dir, 'ped2', 'pred', 'ped2')
    optima_threshold = optimalThreshold(SSIM_values_expand, label_expand)
    print(optima_threshold)