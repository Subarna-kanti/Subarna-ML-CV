#import necessary packages
import cv2
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt


#decalring a class to extract features
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

path = '/home/subarna/Documents/ml_notebooks/assg_1_part2_data'
filenames = glob.glob(path + "/*.jpeg")
dfs = []
for imagePath in filenames:
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
     
    # describe the image
    features = cd.describe(image)
 
    # write the features to file
    features = [str(f) for f in features]
    
    #describing a certain column to specify whether th egiven image is either face or night or landscape
    #(labelling of image)
    category = imagePath.split('.')[0]
    if category == '/home/subarna/Documents/ml_notebooks/assg_1_part2_data/face':
        x = 0
    if category == '/home/subarna/Documents/ml_notebooks/assg_1_part2_data/night':
        x = 1
    if category == '/home/subarna/Documents/ml_notebooks/assg_1_part2_data/landscape':
        x = 2
        
    feat = (imageID,x,",".join(features))
    dfs.append(feat)
df = pd.DataFrame(dfs)  #making dataframes
print(df)

df1 = df[2].str.split(",",expand = True)

del df[2]
for i in range(2,1442):
    df[i] = df1[i-2]

x_train = df1
x_train #training dataframe of features

y_train = df[1]
y_train #training dataframe of outcomes as per the given features

#using knn classifier to classify object
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3)   
knn.fit(x_train, y_train)

while(5):
    print("enter the file name of image:\n")
    img = cv2.imread(input()) #test image
    
    #making test datframe
    dfs = []
    features = cd.describe(img)
    features = [str(f) for f in features]  
    feat = (",".join(features))
    dfs.append(feat)
    test = pd.DataFrame(dfs)
    x_test = test[0].str.split(",",expand = True)
    y_test = knn.predict(x_test)
    
    #displaying results
    if y_test == [0]:
        print('face')
    if y_test == [1]:
        print('night')
    if y_test == [2]:
        print('landscape')
        
    #displaying accuracy
    print(knn.score(x_test, y_test))
    
    #asking for next image
    print("do you want another classification:  yes/no")
    y = input()
    if y == "yes":
        continue  #again new image
    else:
        break #break the loop

