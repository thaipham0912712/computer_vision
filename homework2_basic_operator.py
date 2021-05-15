# Homework1
# subject: Computer vision
# Student: Pham Ngoc Thai 1981105
#-------------------------------------------------------------------------------
# Read image to Matlab
# Show the image to figure
# Extract R,G,B channel
# Show original image, R,G,B channel in the same figure
# Crop image
# Find maximum value of pixel
# Find the position(index) of maximum value
# Find minimum value of pixel
# Find the position of minimum value
# Increase the image intensity linear (multiply 2 times darker and brighter)
# Subtract two images/find the difference of two images (without alignment)
# Subtract two images/find the difference of two images (with alignment) ***

import cv2
import numpy as np
from matplotlib import pyplot as plt 

### Read image to Matlab
img = cv2.imread("sample1.png")

### Show the image to figure
cv2.imshow("sample image",img)

### Extract R,G,B channel
r, b, g = cv2.split(img)

### Show original image, R,G,B channel in the same figure
# create figure 
fig = plt.figure(figsize=(10, 7)) 
  
# setting values to rows and column variables 
rows = 2
columns = 2

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(img) 
plt.axis('off') 
plt.title("origin") 

# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(r) 
plt.axis('off') 
plt.title("red") 

# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(b) 
plt.axis('off') 
plt.title("blue") 

# Adds a subplot at the 4th position 
fig.add_subplot(rows, columns, 4) 
  
# showing image 
plt.imshow(g) 
plt.axis('off') 
plt.title("green") 

# cv2.imshow('concatenated_Hori',img_concate_Hori)
# cv2.imshow('concatenated_Verti',img_concate_Verti)
### Crop image
y=0
x=0
h=100
w=200
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)

### Find maximum value of pixel
# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Apply a Gaussian blur to the image
# gray_img = cv2.GaussianBlur(gray_img, ())
a = np.array(gray_img)
brightest = a.max()
print("Maximum value of pixel:", brightest)
# Find the position(index) of maximum value
print("Index of brightest pixel:", np.unravel_index(a.argmax(), a.shape))
# Find minimum value of pixel
darknest = a.min()
print("Minimum value of pixel:",darknest)
# Find the position of minimum value
print("Index of darknest pixel:", np.unravel_index(a.argmin(), a.shape))
### Increase the image intensity linear (multiply 2 times darker and brighter)
alpha=1
beta=20
new_image=cv2.addWeighted(img, alpha,np.zeros(img.shape, img.dtype),0,beta)
cv2.imshow("Increased img", new_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


