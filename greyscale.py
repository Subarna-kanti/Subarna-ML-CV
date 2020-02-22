#importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading image
a = input("enter the file name of image: ")
img = cv2.imread(a)
plt.figure(figsize = (20,8))
plt.imshow(img)

#function for conversion of image
def gray_convert(img):
    grayvalue = 0.1140*img[:,:,2] + 0.5870*img[:,:,1] + 0.2989*img[:,:,0]
    gray_img = grayvalue.astype(np.uint8)
    return gray_img
gray = gray_convert(img)

#displaying and saving figure
plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(gray,cmap = 'gray')
plt.savefig("output.png")

