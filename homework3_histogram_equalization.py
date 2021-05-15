# Homework 2: Histogram equalization
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Read an image, show image
# Convert to gray scale
# Plot image histogram
# Implement Histogram equalization
# Implement Adaptive Histogram equalization
# Implement Histogram matching

import sys
import numpy as np
import cv2

bins = np.arange(256).reshape(256,1)
def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv2.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


def main():
    # Read an image, show image
    img = cv2.imread("sample1.png")
    cv2.imshow("sample image",img)
    # Convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Plot image histogram
    lines = hist_lines(gray_img)
    cv2.imshow('histogram',lines)
    cv2.imshow('image',gray_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Implement Histogram equalization
# Implement Adaptive Histogram equalization
# Implement Histogram matching

if __name__ == "__main__":
    main()