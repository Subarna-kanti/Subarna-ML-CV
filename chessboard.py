#making chessboard as per the dimension required

import numpy as np
import matplotlib.pyplot as plt

n = int(input())
chessboard = np.zeros((n,n))

chessboard[0::2,1::2] = 1
chessboard[1::2,0::2] = 1

plt.imshow(chessboard,cmap = 'binary')
plt.savefig("output")
