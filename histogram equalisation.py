#importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading image
img = cv2.imread("hill.jpg")

plt.figure(figsize = (20,8))
plt.imshow(img)

#Grayscale conversion
img0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.figure(figsize = (20,8))
plt.imshow(img0, cmap = 'gray')

#applying histogram equalization
equ = cv2.equalizeHist(img0)

plt.figure(figsize = (20,8))
plt.imshow(equ, cmap = 'gray')

#stacking image
res = np.hstack((img0,equ))

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res, cmap = 'gray')
plt.savefig("output.png")

