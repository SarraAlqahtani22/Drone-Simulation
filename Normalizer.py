# Produce NN-like output for information map
import numpy as np
import matplotlib.pyplot as plt
import cv2

box_pixel = 10 ** 2

# choose the output to be normalized
filename = 'GT_Info_Convoluted'
myMap0 = np.load(filename + '.npy')
cv2.imwrite(filename + '.png', np.array(myMap0, dtype=np.uint8))
print('# of true non-empty block: ', len(np.argwhere(myMap0 > 0)))

filename = 'wFakeObjects_Info_Convoluted+s&p+Gaussian'
myMap1 = np.load(filename + '.npy')
cv2.imwrite(filename + '.png', np.array(myMap1, dtype=np.uint8))

# The two lines below defines the transformation algorithm
map2 = (myMap0 - box_pixel / 2) / 9
Sig_y = np.exp(map2) / (np.exp(map2) + 0.035) + 0.06
print('#number of significant block(GC): ', len(np.argwhere(Sig_y > 0.6)))

map2 = (myMap1 - box_pixel / 2) / 9
Sig_y = np.exp(map2) / (np.exp(map2) + 0.035) + 0.06
print('#number of significant block(blurred): ', len(np.argwhere(Sig_y > 0.6)))

np.save(filename + '+Sig+Norm', Sig_y)
plt.hist(Sig_y, bins=10)
plt.show()
