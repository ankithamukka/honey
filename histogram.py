import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('/home/ankitha-mukka/Desktop/experiments/r.jpg')
cv.imwrite("/home/ankitha-mukka/Desktop/experiments/anu.jpg",img)
assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()