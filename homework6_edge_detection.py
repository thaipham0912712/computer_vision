# Homework 6: Edge detection
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Import an image to Matlab/python/C++
# Extract the edge using Robert, Prewitt, Sobel operator
# Extract the edge using Canny method and compare with the previous methods mentioned above
# Hough transform:
# Find the straight line in the edge image extracted by Canny method
# Find the circles in the edge image extracted by Canny method

import sys
import numpy as np
import cv2

# CannyEdge Detection k is the Gaussian kernel size, t1, t2 is the threshold size
def Canny(image,k,t1,t2):
    img = cv2.GaussianBlur(image, (k, k), 0)
    canny = cv2.Canny(img, t1, t2)
    return canny
    
def main():
    # Read an image, show image
    img = cv2.imread("Elizabeth_Olsen.jpg")
    original_image = cv2.resize(img, (600, 900))
    original_image1 = cv2.resize(img, (600, 900))
    cv2.imshow("Elizabeth Olsen",original_image)
    # Convert to BW image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Extract the edge using Robert, Prewitt, Sobel operator
    image = cv2.resize(gray_img,(800,800))
    # Custom convolution kernel
    # Roberts edge operator
    kernel_Roberts_x = np.array([
        [1, 0],
        [0, -1]
        ])
    kernel_Roberts_y = np.array([
        [0, -1],
        [1, 0]
        ])
    # Sobel edge operator
    kernel_Sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    kernel_Sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])
    # Prewitt edge operator
    kernel_Prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])
    kernel_Prewitt_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]])
    # convolution
    output_1 = cv2.filter2D(image, -1, kernel_Roberts_x)
    output_2 = cv2.filter2D(image, -1, kernel_Sobel_x)
    output_3 = cv2.filter2D(image, -1, kernel_Prewitt_x)
    # Show sharpening effect
    image = cv2.resize(image, (600, 900))
    output_1 = cv2.resize(output_1, (600, 900))
    output_2 = cv2.resize(output_2, (600, 900))
    output_3 = cv2.resize(output_3, (600, 900))
    cv2.imshow('Original Image', image)
    cv2.imshow('Roberts Image', output_1)
    cv2.imshow('Sobel Image', output_2)
    cv2.imshow('Prewitt Image', output_3)
    # Extract the edge using Canny method and compare with the previous methods mentioned above
    output_5 = Canny(image,3,50,150)
    cv2.imshow('Canny Image', output_5)
    # Hough transform:
    #   - Find the straight line in the edge image extracted by Canny method
    # This returns an array of r and theta values
    lines = cv2.HoughLines(output_5,1,np.pi/180, 200)
    
    # The below for loop runs till r and theta values 
    # are in the range of the 2d array
    for r,theta in lines[0]:
        
        # Stores the value of cos(theta) in a
        a = np.cos(theta)
    
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
        
        # x0 stores the value rcos(theta)
        x0 = a*r
        
        # y0 stores the value rsin(theta)
        y0 = b*r
        
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
        
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
        
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
        
        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be 
        #drawn. In this case, it is red. 
        cv2.line(original_image,(x1,y1), (x2,y2), (0,0,255),2)
    cv2.imshow('Straight line in edge image', original_image)
    #   - Find the circles in the edge image extracted by Canny method  
    # detect circles in the image
    circles = cv2.HoughCircles(output_5, cv2.HOUGH_GRADIENT, 1.2, 100)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        cv2.imshow("circles in edge image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()