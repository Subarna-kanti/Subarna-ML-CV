#imprting necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading image
img = cv2.imread("flow.jpg")
plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(img)

grid_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(grid_RGB)

#Converting to HSV
grid_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(grid_HSV)

#parameters for mask
lower = np.array([60,150,50])
upper = np.array([90,255,255])
#mask creation
mask = cv2.inRange(grid_HSV,lower,upper)

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(mask)

#convertign the masked result to visible region(RGB)
res = cv2.bitwise_and(grid_RGB,grid_RGB,mask = mask)

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res)
plt.savefig("grass")

#parameters for mask creation for flowers
lower_fl = np.array([0,0,0])
upper_fl = np.array([60,255,255])
mask_fl = cv2.inRange(grid_HSV,lower_fl,upper_fl)

lower_fl1 = np.array([90,0,0])
upper_fl1 = np.array([180,255,255])
mask_fl1 = cv2.inRange(grid_HSV,lower_fl1,upper_fl1)

mask_main = cv2.bitwise_or(mask_fl,mask_fl1)

res_fl = cv2.bitwise_and(grid_RGB,grid_RGB,mask = mask_main)

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res_fl)
plt.savefig("flowers")

