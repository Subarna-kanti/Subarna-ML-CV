import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("my_image.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


plt.figure(figsize = (20,8))
plt.imshow(img, cmap = 'gray')

img_norm = img/255
mean = img_norm.mean()
std = np.std(img_norm)
var = np.var(img_norm)
im0 = (img_norm - mean)/var
im0 = im0 + 120

res = np.hstack((im0,img))

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res, cmap = 'gray')
plt.savefig("output.png")

