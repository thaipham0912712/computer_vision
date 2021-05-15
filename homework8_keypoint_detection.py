# Homework 8: Keypoint detection
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Import an image to Matlab/python/C++
# Extract these features:
# - Laplacian detector
# - Determinant of Hessian detector
# - Harris detector
# - FAST detector
# - Plot superimpose on the image

import numpy as np
from matplotlib import pyplot as plt
import cv2

def main():
    # Read an image, show image
    img_rgb = cv2.imread('Elizabeth_Olsen.png')
    cv2.imshow("Original photo", img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()