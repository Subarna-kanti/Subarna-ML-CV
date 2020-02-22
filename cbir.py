# import the necessary packages
import numpy as np
import cv2
import glob
import pandas as pd
from matplotlib import pyplot as plt

class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

    # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]

        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

path ='/home/subarna/Documents/ml_notebooks/dataset'
filenames = glob.glob(path + "/*.png")
dfs = []
for imagePath in filenames:
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
     
    # describe the image
    features = cd.describe(image)
 
    # write the features to file
    features = [str(f) for f in features]
    feat = (imageID,",".join(features))
    dfs.append(feat)
df = pd.DataFrame(dfs)

df1 = df[1].str.split(",",expand = True)
del df[1]

for i in range(1,1441):
    df[i] = df1[i-1]

tar_img = cv2.imread("/home/subarna/Documents/ml_notebooks/dataset/"+input())

target_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# describe the image
features = cd.describe(target_img)
df_feat = pd.DataFrame(features)
df_feat  = df_feat.T
df1 = df1.astype(float)
df_feat = df_feat.astype(float)

eps = 1e-10
dfs = []
for j in range(0,805):
    d = 0.5 * np.sum([((float(df1[j:j+1][i]) - float(df_feat[i])) ** 2) / (float(df1[j:j+1][i]) + float(df_feat[i]) + eps)
        for i in range(0,1440)])
    print(d)
    dfs.append(d)
data = pd.DataFrame(dfs)
data.to_csv("out.csv",index = True)
df[1] = data
df.sort_values(1,ascending = True,inplace = True)
fd = df.head(10)
a = fd[0].values.tolist()
for i in range(0,10):
    img= cv2.imread("/home/subarna/Documents/ml_notebooks/dataset/"+str(a[i]))
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize = (20,8))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig("want"+str(i))

