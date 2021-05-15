# Homework 3: Image segmentation
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Read (rice.jpeg) in matlab image data
# Find threshold
# Convert to BW image
# Count the number of rice
# Remove small regions (noise region)
# Labelling for eacpythonh region
# Show the color of regions
# Bounding box each region

import sys
import numpy as np
import cv2

def main():
    # Read an image, show image
    img = cv2.imread("rice.jpg")
    cv2.imshow("rices image",img)
    # Convert to BW image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove small regions (noise region)
    blur = cv2.GaussianBlur(gray_img,(5,5),0)
    # Find threshold
    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_gauss_threshold, image_result = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print("Obtained threshold: ", otsu_gauss_threshold)
    cv2.imshow("image_result",image_result)
    # Count the number of rice
    num_labels, labels = cv2.connectedComponents(image_result)
    print("Number of rice:", num_labels)
    # Labelling for each region
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # Show the color of regions
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    cv2.imshow("labeled result", labeled_img)

    # Bounding box each region
    contours, hier = cv2.findContours(image_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))

        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)

    print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imshow("contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()