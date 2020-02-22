import cv2
import numpy as np
from matplotlib import pyplot as plt

img0 = cv2.imread('rose_sun.jpg')
img1 = cv2.imread('rose.jpg')

img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
plt.figure(figsize = (20,8))
plt.imshow(img0)

plt.figure(figsize = (20,8))
plt.imshow(img1)

res = np.hstack((img0,img1))
plt.figure(figsize = (20,8))
plt.imshow(res)
img0_h = cv2.cvtColor(img0,cv2.COLOR_RGB2HSV)
img1_h = cv2.cvtColor(img1,cv2.COLOR_RGB2HSV)

arr = img0_h - img1_h

H = img0_h.copy()
H[:,:,2] =120
H
new0 = cv2.cvtColor(H,cv2.COLOR_HSV2RGB)
H1 = img1_h.copy()
H1[:,:,2] =120
H1
new1 = cv2.cvtColor(H1,cv2.COLOR_HSV2RGB)
res_new = np.hstack((new0,new1))
plt.figure(figsize = (20,8))
plt.imshow(res_new) #V = 120

H = img0_h.copy()
H[:,:,2] =255
H
new0 = cv2.cvtColor(H,cv2.COLOR_HSV2RGB)
H1 = img1_h.copy()
H1[:,:,2] =255
H1
new1 = cv2.cvtColor(H1,cv2.COLOR_HSV2RGB)
res_new = np.hstack((new0,new1))
plt.figure(figsize = (20,8))
plt.imshow(res_new) #V = 120

plt.figure(figsize = (20,8))
plt.imshow(new0) #V = 255

plt.figure(figsize = (20,8))
plt.imshow(new0) #V = 0

plt.figure(figsize = (20,8))
plt.imshow(new0) #S = 0

plt.figure(figsize = (20,8))
plt.imshow(new0) #S = 255

plt.figure(figsize = (20,8))
plt.imshow(new0) #H = 0

plt.figure(figsize = (20,8))
plt.imshow(new0) #H = 180

plt.figure(figsize = (20,8))
plt.imshow(new0) #H = 90

H = img0_h.copy()
H[:,:,0] =90
H
new0 = cv2.cvtColor(H,cv2.COLOR_HSV2RGB)

H = img0_h.copy()
H[:,:,0] =90
H
new0 = cv2.cvtColor(H,cv2.COLOR_HSV2RGB)

