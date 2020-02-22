#import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load the image
img = cv2.imread('spot.jpg')


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
np.log(1+np.max(img))

# Apply log transform
img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255
img_log = np.array(img_log,dtype=np.uint8)
res_new = np.hstack((img,img_log))

plt.figure(figsize = (20,8))
plt.axis("off")
plt.imshow(res_new)
plt.savefig("output.png")

