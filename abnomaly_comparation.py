import argparse
import cv2 as cv
from Image_Similarity import CompareHistogram as ch
from Image_Similarity import CompareFeatures as cf
from Image_Similarity import Image_Diff as id

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", type=str,
                    default='./Image_Similarity/Dataset/125.jpg', help="first input image")
    ap.add_argument("-s", "--second", type=str,
                    default='./Image_Similarity/Dataset/125_blur.jpg', help="second")
    ap.add_argument("-m", "--mode", type=int, default=2, help="Mode")
    args = vars(ap.parse_args())

    # load the two input images
    imageA = cv.imread(args["first"])
    imageB = cv.imread(args["second"])

    # load mode of program
    mode = int(args["mode"])

    if mode == 0:
        ch.compareHistogram(imageA, imageB)
    elif mode == 1:
        cf.compareSIFT(imageA, imageB)
    elif mode == 2:
        id.image_differences(imageA, imageB)

    cv.waitKey(0)
