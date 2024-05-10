import cv2
import numpy as np

bounding_box_image = cv2.imread('PedestrianRectangles/1/grim.pgm')

edges = cv2.Canny(bounding_box_image, 50, 100)  # apertureSize=3

cv2.imshow('edge', edges)
cv2.waitKey(0)

lines = cv2.HoughLinesP(edges, rho=0.5, theta=1 * np.pi / 180, 
threshold=100, minLineLength=100, maxLineGap=50)

# print(len(lines))

for i in lines:
    for x1, y1, x2, y2 in i:
        # print(x1, y1, x2, y2)
        cv2.line(bounding_box_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('houghlines5.jpg', bounding_box_image)


