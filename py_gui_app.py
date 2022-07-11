
from email.policy import default
from skimage.metrics import structural_similarity
import imutils
import cv2
import tkinter as tk
import numpy as np
from yaml import load
from image_similarity import ImageDifference as id
import os

def image_differences_subtract(imageA, imageB, pred_score, threshold):
    print("score: ", pred_score)
    imgA_clone = imageA.copy()
    imgB_clone = imageB.copy()
    if pred_score < threshold:  # if frame has score lower than threshold
        # convert RGB images to grayscale images
        grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        diff = cv2.subtract(grayA, grayB)
        cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
        #cv2.waitKey(0)

        diff = (diff * 255).astype("uint8")
        diff = cv2.GaussianBlur(diff, (7, 7), sigmaX=3, sigmaY=3)
        
        #print("SSIM: {}".format(score))
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        my_threshold = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        threshA = cv2.threshold(diff, 0, 255, my_threshold)
        thresh = threshA[1]
        cv2.imshow("Thresholed", cv2.resize(thresh, None, fx=1, fy=1))
        #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #thresh = cv2.threshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            # Remove regions in-significant regions
            #if w < 5 or h < 20:
                #continue
            cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
         
        # show the output images
        cv2.imshow("Detect regions", cv2.resize(imgA_clone, None, fx=1, fy=1))
        #cv2.imwrite("Result_Original.png", imageA)
        #cv2.imshow("Pred frame compared", cv2.resize(imgB_clone, None, fx=1, fy=1))
        cv2.waitKey(0)
        #cv2.imwrite("Result_Modified.png", imageB)
        #cv2.imshow("Diff", cv2.resize(diff, None, fx=1, fy=1))
        #cv2.imwrite("Result_Diff.png", diff)
        #cv2.imshow("Thresh", cv2.resize(thresh, None, fx=1, fy=1))
        #cv2.imwrite("Result_Thresh.png", thresh)
        #cv2.waitKey(0)
        return imgA_clone, imgB_clone
    else:
        #cv2.imshow("Test frame compared", cv2.resize(imageA, None, fx=1, fy=1))
        #cv2.imwrite("Result_Original.png", imageA)
        #cv2.imshow("Pred frame compared", cv2.resize(imageB, None, fx=1, fy=1))
        return imgA_clone, imgB_clone

def image_differences_type2(imageA, imageB, pred_score, threshold):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = structural_similarity(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
            cv2.imshow("d BINARY_INV", cv2.resize(diff, None, fx=1, fy=1))
            print("SSIM: {}".format(score))
            
            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh_type = cv2.THRESH_BINARY_INV
            threshA = cv2.threshold(diff, 0, 255, thresh_type)
            thresh = threshA[1]
            cv2.imshow("t BINARY_INV", cv2.resize(thresh, None, fx=1, fy=1))
            
            #thresh = cv2.threshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 5 or h < 10:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(imgB_clone, text="w:{width}, h:{heigth}".format(width=w, heigth=h), 
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1, org=(10, 10))
            #cv2.imshow("Test frame compared", cv2.resize(imgA_clone, None, fx=1, fy=1))
            #cv2.imwrite("Result_Original.png", imageA)
            cv2.imshow("BINARY_INV", cv2.resize(imgB_clone, None, fx=1, fy=1))

            return imgA_clone, imgB_clone, thresh
        else:
            blank_image = np.zeros((256,256,3), np.uint8)
            return imgA_clone, imgB_clone, blank_image

def image_differences_type1(test_image, predicted_image, pred_score, threshold):
    print("score: ", pred_score)
    test_clone = test_image.copy()
    pred_clone = predicted_image.copy()

    # convert RGB images to grayscale images
    grayA = cv2.cvtColor(test_clone, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(pred_clone, cv2.COLOR_BGR2GRAY)        
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (SSIM_score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
    #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
    print("SSIM: {}".format(SSIM_score))

    if pred_score < threshold:  # check if frame has score lower than threshold
        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        # both of these settings (BINARY_INV and OTSU) are applied at the same time using the vertical bar
    
        # Firstly, we need to separate foreground and background
        foreground_extracted = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
        # Then, inverse change pixel intensity for visualizing
        inverse_foreground = cv2.threshold(foreground_extracted, 0, 255, cv2.THRESH_BINARY_INV)
        # Finally, get the thresholded image
        thesholded_img = inverse_foreground[1]


        # Post-Process data
        #kernel_2 = np.ones((2, 2), np.uint8)
        #kernel_3 = np.ones((3, 3), np.uint8)
        #kernel_4 = np.ones((4, 4), np.uint8)
        kernel_5 = np.ones((5, 5), np.uint8)
        kernel_7 = np.ones((7, 7), np.uint8)
        kernel_11 = np.ones((11, 11), np.uint8)
        #Closing_img = cv2.morphologyEx(thesholded_img, cv2.MORPH_CLOSE, kernel=kernel_4, iterations=1)  # dilate -> erode 
        Opening_img = cv2.morphologyEx(thesholded_img, cv2.MORPH_OPEN, kernel=kernel_5, iterations=1)  # erode -> dilate
        #Eroded_img = cv2.erode(Opening_img, kernel=kernel_7, iterations=1)

        #complete_process_img = cv2.erode(Closing_img, kernel=kernel_2, iterations=1)
        complete_process_img = Opening_img
        # Next step, we find the contours of thresholded image
        cnts = cv2.findContours(complete_process_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours in order to show bbox as well as eliminate small regions 
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)

            # Remove regions in-significant regions
            if w < 17 or h < 17:
                continue

            cv2.rectangle(test_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(pred_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return test_clone, pred_clone, complete_process_img, diff, SSIM_score
    else:
        theshold_blank_image = np.zeros((256, 256, 3), np.uint8)
        diff_blank_img = np.zeros((256, 256, 3), np.uint8)
        diff_blank_img += 255
        return test_clone, pred_clone, theshold_blank_image, diff_blank_img, SSIM_score

def image_differences_type1_unfilter(imageA, imageB, pred_score, threshold):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = structural_similarity(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            #diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
            cv2.imshow("du BINARY_INV_OTSU", cv2.resize(diff, None, fx=1, fy=1))
            print("SSIM: {}".format(score))
            
            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh_type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            threshA = cv2.threshold(diff, 255, 255, thresh_type)
            thresh = threshA[1]
            cv2.imshow("tu BINARY_INV_OTSU", cv2.resize(thresh, None, fx=1, fy=1))
            
            #thresh = cv2.threshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w <= 5 or h < 12:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if w > 10 or h > 20:
                    pass
                else:
                    cv2.putText(imgA_clone, text="w:{width}, h:{heigth}".format(width=w, heigth=h), 
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1, org=(10, 10))
            #cv2.imshow("Test frame compared", cv2.resize(imgA_clone, None, fx=1, fy=1))
            #cv2.imwrite("Result_Original.png", imageA)
            cv2.imshow("u BINARY_INV_OTSU", cv2.resize(imgA_clone, None, fx=1, fy=1))
            return imgA_clone, imgB_clone, thresh
        else:
            blank_image = np.zeros((256,256,3), np.uint8)
            return imgA_clone, imgB_clone, blank_image

# lr1 = 15e-4
# lr2 = 2e-3
# print(lr1)
# print(lr2)
#true_index_of_test_frame = i + self.frame_sequence_length * test_video_index[i+map_index*4]

def load_pixelLabel_frames(dataset_type='ped2'):
    label_input_path = []
    label_dir = []
    label_dir_distinct = []
    cur_path = './dataset/' + dataset_type + '/testing/labels'
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

pixel_labels = load_pixelLabel_frames

ID = id.ImageDifference()
test_frame = cv2.imread('./dataset/ped2/testing/frames/01/120.jpg')
test_frame = cv2.resize(test_frame, (256, 256))
predicted_frame = cv2.imread('./dataset/ped2/output/frames/0116.jpg')
predicted_frame = cv2.resize(predicted_frame, (256, 256))
cv2.imshow("test frame", cv2.resize(test_frame, (256, 256)))
cv2.imshow("predict frame", cv2.resize(predicted_frame, (256, 256)))
#cv2.waitKey(0)
#image_differences_subtract(imageA, imageB, 0.7, 0.8)
#image_differences_type2(imageA, imageB, 0.7, 0.8)
#image_differences_type1_unfilter(imageA, imageB, 0.7, 0.8)
    # *** PIXEL LEVEL
regions_test, regions_pred, complete_process_img, SSIM_diff_img, SSIM_score = ID.image_differences(
    test_frame, predicted_frame, 0.6, 0.8)
cv2.imshow("regions_test", cv2.resize(regions_test, (256, 256)))
cv2.imshow("complete_process_img", cv2.resize(SSIM_diff_img, (256, 256)))
cv2.waitKey(0)

