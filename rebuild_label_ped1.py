from scipy.io import loadmat
import h5py
import numpy as np
anomalyFrame = [[[]]]
# video 1
anomalyFrame[0][0].append(65)
anomalyFrame[0][0].append(148)

# video: 2
anomalyFrame.append([[60, 165]])

# video: 3
anomalyFrame.append([[95, 200]])

# video: 4
anomalyFrame.append([[35, 165]])

# video: 5
anomalyFrame.append([[10, 82]])
anomalyFrame[4].append([120, 200])

# video: 6
anomalyFrame.append([[1, 92]])
anomalyFrame[5].append([115, 200])

# video: 7
anomalyFrame.append([[1, 168]])

# video: 8
anomalyFrame.append([[1, 85]])

# video: 9
anomalyFrame.append([[1, 50]])

# video: 10
anomalyFrame.append([[1, 130]])

# video: 11
anomalyFrame.append([[74, 159]])

# video: 12
anomalyFrame.append([[140, 200]])

# video: 13
anomalyFrame.append([[1, 156]])

# video: 14
anomalyFrame.append([[1, 200]])

# video: 15
anomalyFrame.append([[140, 200]])

# video: 16
anomalyFrame.append([[134, 195]])

# video: 17
anomalyFrame.append([[1, 42]])

# video: 18
anomalyFrame.append([[54, 119]])

# video: 19
anomalyFrame.append([[64, 136]])

# video: 20
anomalyFrame.append([[49, 165]])

# video: 21
anomalyFrame.append([[45, 200]])

# video: 22
anomalyFrame.append([[20, 105]])

# video: 23
anomalyFrame.append([[8, 160]])

# video: 24
anomalyFrame.append([[50, 169]])

# video: 25
anomalyFrame.append([[52, 32]])

# video: 26
anomalyFrame.append([[80, 140]])

# video: 27
anomalyFrame.append([[15, 120]])

# video: 28
anomalyFrame.append([[115, 200]])

# video: 29
anomalyFrame.append([[1, 8]])
anomalyFrame[28].append([50, 111])

# video: 30
anomalyFrame.append([[175, 200]])

# video: 31
anomalyFrame.append([[1, 155]])

# video: 32
anomalyFrame.append([[1, 50]])
anomalyFrame[31].append([65, 115])

# video: 33
anomalyFrame.append([[12, 165]])

# video: 34
anomalyFrame.append([[12, 110]])

# video: 35
anomalyFrame.append([[92, 200]])

# video: 36
anomalyFrame.append([[25, 105]])
print('Done')

arr = np.zeros(0, np.int64)

for videoIndex, rangeAno in enumerate(anomalyFrame):
    arrItem = np.zeros(200, np.int64)
    for index in range(200):
        start = rangeAno[0][0]
        end = rangeAno[0][1]
        if index > start - 5 and index < end + 5:
            arrItem[index] = 1
        else:
            arrItem[index] = 0

        if index == 0:
            # print('Video:', videoIndex+1, 'start:', start, 'end:', end)
            print('\n# video:', videoIndex+1)
            print('a.append([['+str(start)+', '+str(end)+']])')

        if index >= end + 5 and len(rangeAno) == 2:
            start = rangeAno[1][0]
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
