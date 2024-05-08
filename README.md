HISTOGRAM

A histogram is the graphical representation of data where data is grouped into continuous number ranges and each range corresponds to a vertical bar.

The horizontal axis displays the number range.
the vertical axis (frequency) represents the amount of data that is present in each range.

The number ranges depend upon the data that is being used.


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

# libraries uses

# import numpy as np: 
Imports the NumPy library, usually aliased as np, which is commonly used for numerical computations in Python.

# import cv2 as cv: 
Imports the OpenCV library, usually aliased as cv, which is a popular library for computer vision tasks.

# from matplotlib import pyplot as plt: 
Imports the pyplot module from the Matplotlib library, usually aliased as plt, which is used for data visualization.
```
img = cv.imread('/home/ankitha-mukka/Desktop/experiments/r.jpg'): Reads an image file named "r.jpg" located at the specified path using OpenCV's imread function. The image is stored in the variable img.

cv.imwrite("/home/ankitha-mukka/Desktop/experiments/anu.jpg", img): Writes the image img to another file named "anu.jpg" at the specified path.
```

assert img is not None, "file could not be read, check with os.path.exists()": Checks if the image was successfully read. If the image is None, it raises an AssertionError with the message "file could not b




