## Iteration


## Exmple program for first 10 numbers:

num = list(range(10))

This line creates a list called num containing numbers from 0 to 9 using the range() function and then converting it into a list.

previousNum = 0


This initializes a variable called previousNum to 0. This variable will be used to keep track of the previous number in each iteration of the loop.

for i in num:

This is a loop that iterates through each element in the list num.

sum = previousNum + i

This line calculates the sum of the current number (i) and the previous number (previousNum) and assigns it to a variable named sum.


print('Current Number ' + str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))


This line prints the current number (i), the previous number (previousNum), and their sum (sum) in a formatted string.


previousNum = i


This line updates the previousNum variable to the current number (i) for the next iteration of the loop.


So, when you run this code, it will output the current number, the previous number, and their sum for each number in the num list, effectively showing the cumulative sum of numbers from 0 to 9.

## output

Current Number: 1, Previous Number: 0, Sum: 1

Current Number: 2, Previous Number: 1, Sum: 3

Current Number: 3, Previous Number: 2, Sum: 5

Current Number: 4, Previous Number: 3, Sum: 7

Current Number: 5, Previous Number: 4, Sum: 9

Current Number: 6, Previous Number: 5, Sum: 11

Current Number: 7, Previous Number: 6, Sum: 13

Current Number: 8, Previous Number: 7, Sum: 15

Current Number: 9, Previous Number: 8, Sum: 17

Current Number: 10, Previous Number: 9, Sum: 19

## histogram
A histogram is a type of chart that shows the frequency distribution of data points across a continuous range of numerical values. The values are grouped into bin or buckets that are arranged in consecutive order along the horizontal x-axis at the bottom of the chart. Each bin is represented by a vertical bar that sits on the x-axis and extends upward to indicate the number of data points within that bin.

1.Import Libraries:

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

2.Read Image:
 
img = cv.imread('/home/ankitha-mukka/Desktop/experiments/r.jpg')

3.Write Image:

cv.imwrite("/home/ankitha-mukka/Desktop/experiments/anu.jpg",img)

4.Assertion Check:

assert img is not None, "file could not be read, check with os.path.exists()"

5.Calculate Histogram:

color = ('b','g','r')

for i,col in enumerate(color):

 histr = cv.calcHist([img],[i],None,[256],[0,256])
 
 6.Plot Histogram:
 
 plt.plot(histr,color = col)
 
 plt.xlim([0,256])
 
plt.show()

## input:

![r](https://github.com/ankithamukka/honey/assets/169052286/d6992a29-87f4-46c0-9e6b-dc93fd27b976)

## output:

![histogram](https://github.com/ankithamukka/honey/assets/169052286/9d2a01c1-fa50-4c5f-b78b-2d592ddbb92d)

## bounding boxes

A bounding box is the smallest rectangle with vertical and horizontal sides that completely surrounds an object. All portions of the object lie within the bounding box. The bounding box of a selected object is indicated by selection handles.

1.import libraries:

    os: This module provides functions for interacting with the operating system, such as creating directories.
    
    csv: This module facilitates reading and writing CSV files.
    
    PIL.Image and PIL.ImageDraw: These classes from the Python Imaging Library (PIL) allow for image manipulation, including drawing on images.

2.reading file paths:

    csv_file: Path to the CSV file containing bounding box information.
    
    image_dir: Directory containing the images.
    
3.Create Output Directory:
    
    output_dir: Directory where images with bounding boxes will be saved.
    
os.makedirs(output_dir, exist_ok=True)

4.defining functions:

draw_boxes(image, boxes): Draws red rectangles around the bounding boxes on the input image.

def draw_boxes(image, boxes):

    draw = ImageDraw.Draw(image)
    
    for box in boxes:
    
        left = int(box['left'])
        
        top = int(box['top'])
        
        right = int(box['right'])
        
        bottom = int(box['bottom'])
        
        draw.rectangle([left, top, right, bottom], outline="red")
        
    return image

crop_image(image, boxes): Crops regions of interest from the image based on bounding box coordinates and returns a list of cropped images

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
    
5.Processing CSV File:

with open(csv_file, 'r') as file:

6.Iterates over each row in the CSV file. For each row:

    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))


