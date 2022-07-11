import numpy as np
import h5py
from scipy.io import loadmat
data = loadmat(
    "./dataset/ped1/ped1.mat")
arr = np.zeros(0, np.int64)

for videoIndex, rangeAno in enumerate(data['gt'][0]):
    a = len(rangeAno[0])
    arrItem = np.zeros(200, np.int64)
    for index in range(200):
        start = rangeAno[0][0]
        end = rangeAno[1][0]
        if index > start and index < end:
            arrItem[index] = 1
        else:
            arrItem[index] = 0

        if index == 0:
            # print('Video:', videoIndex+1, 'start:', start, 'end:', end)
            print('\n# video:', videoIndex+1)
            print('a.append([['+str(start)+', '+str(end)+']])')

        if index >= end and len(rangeAno[0]) == 2:
            start = rangeAno[0][1]
            end = rangeAno[1][1]
            if index > start and index < end:
                arrItem[index] = 1
            else:
                arrItem[index] = 0
            if index == start:
                print('a['+str(videoIndex) +
                      '].append(['+str(start)+', '+str(end)+'])')
                # print('Video:', videoIndex+1, 'start:', start, 'end:', end)
    arr = np.concatenate((arr, arrItem), axis=None)
print('Done')
arr = np.array([arr])
np.save('frame_labels_ped1.npy', arr)
