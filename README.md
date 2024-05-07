## crop.py
```
import os
import csv
from PIL import Image, ImageDraw


csv_file = "/home/ankitha-mukka/7622202030987_bounding_box.csv"
image_dir = "/home/ankitha-mukka/7622202030987"
output_dir = "/home/ankitha-mukka/7622202030987_with_boxes"


os.makedirs(output_dir, exist_ok=True)


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image


def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images


with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': 
```

## histogram.py

```
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
```

## iterate.py
```
def print_sum_of_previous_and_current():
    previous_number = 0
    for i in range(1, 11):
        current_number = i
        sum_of_previous_and_current = previous_number + current_number
        print(f"Current Number: {current_number}, Previous Number: {previous_number}, Sum: {sum_of_previous_and_current}")
        previous_number = current_number

print_sum_of_previous_and_current()
```

## videocp.py
```
# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
```

