import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

img = cv2.imread("flow.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test_str = "Hi_my_name_is_SUBARNA_"
res = ''.join(format(ord(i), 'b') for i in test_str)
print(res)

img.shape
plt.figure(figsize = (20,8))
plt.imshow(img)
img_new = img.copy()

#encoding the name(or secret message)

q = 0
r = 0
for i in range (0,len(res)):
    
    s = '{0:08b}'.format(img_new[0,q,r])
    
    if(s[7] == res[i]):
        r = r + 1
        if(r <= 2):
            continue
        else:
            r = 0
            q = q + 1
        continue
    elif(s[7] == '1'):
        img_new[0,q,r] = img_new[0,q,r] - 1
    else:
        img_new[0,q,r] = img_new[0,q,r] + 1
    
    r = r + 1
    
    if(r <= 2):
        continue
    else:
        r = 0
        q = q + 1
plt.figure(figsize = (20,8))
plt.axis("off")
plt.savefig("img_new.png")
plt.imshow(img_new)

#Decoding our secret message

q = 0
r = 0
p = ''
t = ''
u = 0
for i in range(0,301):
    s = '{0:08b}'.format(img_new[0,q,r])
    p = p + s[7]
    r = r + 1
    
    if(u == 6):
        t = t + chr(int(p,2))
        p = ''
        u = -1
    
    u = u + 1
    if(r <= 2):
        continue
    else:
        r = 0
        q = q + 1

print(t)

