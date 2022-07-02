from scipy.io import loadmat
import h5py
import numpy as np
anomalyFrame = [[[]]]
# video 1
anomalyFrame[0][0].append(63)
anomalyFrame[0][0].append(150)

# video: 2
anomalyFrame.append([[60, 172]])

# video: 3
anomalyFrame.append([[85, 200]])

# video: 4
anomalyFrame.append([[8, 168]])

# video: 5
anomalyFrame.append([[6, 88]])
anomalyFrame[4].append([100, 200])

# video: 6
anomalyFrame.append([[1, 98]])
anomalyFrame[5].append([109, 200])

# video: 7
anomalyFrame.append([[1, 175]])

# video: 8
anomalyFrame.append([[1, 93]])

# video: 9
anomalyFrame.append([[1, 55]])

# video: 10
anomalyFrame.append([[1, 155]])

# video: 11
anomalyFrame.append([[60, 161]])

# video: 12
anomalyFrame.append([[125, 200]])

# video: 13
anomalyFrame.append([[1, 200]])

# video: 14
anomalyFrame.append([[1, 200]])

# video: 15
anomalyFrame.append([[147, 200]])

# video: 16
anomalyFrame.append([[125, 195]])

# video: 17
anomalyFrame.append([[1, 45]])

# video: 18
anomalyFrame.append([[44, 120]])

# video: 19
anomalyFrame.append([[58, 138]])

# video: 20
anomalyFrame.append([[45, 169]])

# video: 21
anomalyFrame.append([[40, 200]])

# video: 22
anomalyFrame.append([[9, 107]])

# video: 23
anomalyFrame.append([[6, 165]])

# video: 24
anomalyFrame.append([[44, 171]])

# video: 25
anomalyFrame.append([[46, 133]])

# video: 26
anomalyFrame.append([[76, 140]])

# video: 27
anomalyFrame.append([[10, 120]])

# video: 28
anomalyFrame.append([[113, 200]])

# video: 29
anomalyFrame.append([[1, 13]])
anomalyFrame[28].append([44, 112])

# video: 30
anomalyFrame.append([[175, 200]])

# video: 31
anomalyFrame.append([[1, 166]])

# video: 32
anomalyFrame.append([[1, 52]])
anomalyFrame[31].append([63, 115])

# video: 33
anomalyFrame.append([[6, 165]])

# video: 34
anomalyFrame.append([[10, 114]])

# video: 35
anomalyFrame.append([[86, 200]])

# video: 36
anomalyFrame.append([[22, 107]])
print('Done')

arr = np.zeros(0, np.int64)

for videoIndex, rangeAno in enumerate(anomalyFrame):
    arrItem = np.zeros(200, np.int64)
    for index in range(200):
        start = rangeAno[0][0]
        end = rangeAno[0][1]
        if index > start + 1 and index < end - 1:
            arrItem[index] = 1
        else:
            arrItem[index] = 0

        if index == 0:
            # print('Video:', videoIndex+1, 'start:', start, 'end:', end)
            print('\n# video:', videoIndex+1)
            print('a.append([['+str(start)+', '+str(end)+']])')

        if len(rangeAno) == 2 and index > end:
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
