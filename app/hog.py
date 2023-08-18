import cv2
import numpy as np
def hog_des(img_gray):
# img_gray = cv2.imread('D:\\ImageFeature\\test\\Audi\\41.jpg', 0)
        img_new = cv2.resize(img_gray, (128,128), cv2.INTER_AREA)
        win_size = (128,128)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
        cell_size, num_bins)

        hog_descriptor = hog.compute(img_new)
        # print ('HOG Descriptor:', hog_descriptor)
        return hog_descriptor