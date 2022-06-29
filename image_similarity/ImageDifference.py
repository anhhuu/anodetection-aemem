# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
import tkinter as tk
import numpy as np
from torch import inverse


class ImageDifference():
    def __init__(self) -> None:
        print("")

    def image_differences(self, imageA, imageB, pred_score, threshold):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()

        # convert RGB images to grayscale images
        grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)        
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
        #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
        print("SSIM: {}".format(score))

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

            # Next step, we find the contours of thresholded image
            cnts = cv2.findContours(thesholded_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours in order to show bbox as well as eliminate small regions 
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 20 or h < 20:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return imgA_clone, imgB_clone, thesholded_img, diff, score
        else:
            theshol_blank_image = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return imgA_clone, imgB_clone, theshol_blank_image, diff_blank_img, score

    def image_differences_pixel_label(self, imageA, imageB, pred_score, threshold, pixel_label):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # check if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            #(score, diff) = structural_similarity(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
            
            #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
            #print("SSIM: {}".format(score))

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            # both of these settings (BINARY_INV and OTSU) are applied at the same time using the vertical bar
            #thresh = cv2.threshold(diff, 190, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = np.uint8(pixel_label)
            # find the contours of thresholded image
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 20 or h < 20:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return imgA_clone, imgB_clone, thresh, diff_blank_img

        else:
            theshol_blank_image = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return imgA_clone, imgB_clone, theshol_blank_image, diff_blank_img
