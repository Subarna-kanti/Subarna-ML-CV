#importing necessary library
import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading image
img = cv2.imread('flow_comb.jpg')

#BGR to RGB conversion
RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#displaying input image
plt.figure(figsize = (20,8))
plt.imshow(RGB)

R = RGB.copy()
G = RGB.copy()
B = RGB.copy()

R[:,:,(1,2)] = 0
G[:,:,(0,2)] = 0
B[:,:,(0,1)] = 0

#CMY formula
M = R + B
C = G + B
Y = R + G

#Stacking image
res = np.hstack((C,M,Y))

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res)
plt.savefig("output.png")

