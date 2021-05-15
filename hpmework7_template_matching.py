# Homework 7: Template matching
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Import an image to Matlab/python/C++
# Extract a region in the image as a template
# Using template matching method based on the area correlation to find the location of the template in the image

import numpy as np
from matplotlib import pyplot as plt
import cv2

def main():
    # Read an image, show image
    img_rgb = cv2.imread('super_mario.png')
    cv2.imshow("Original photo", img_rgb)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('question_box_template.jpg',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imshow("Result", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()