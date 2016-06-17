# cavr-mosaic
OpenCV python code for creating a mosaic of images of the seafloor using video from autonomous robots

Goal 1: Reading Individual Frames from Video File
=================================================
With OpenCV it is fairly easy to obtain individual images from a video file.  Here is some code that will do so:
```
#! /usr/bin/env python

import cv2
import numpy as np

video = cv2.VideoCapture("../140723_162428.MPG")
ret, frame = video.read() #frame is the image file created
```
Here is the [documentation](http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get 'Reading and Writing Images and Video') for reading and writing images and Video in OpenCV.

Goal 2: Vision Processing Software
==================================
Training OpenCV to Recognize Digits
-----------------------------------
![alt Learner Image](learner_done2.png)

OpenCV learns how to read characters by using the K-Nearest Neighbors algorithm.  It requires a training file in order to learn.  I created this image using the digits in the NEEMO videos and made alterations to them in order to capture many different possibilities of how they may appear in the video.  I placed all of these digits in one file and used array slicing to break it into individual characters.  When I tested this training file, it provided 100% accuracy, most likely because all of the digits look so similar.  I will soon test it on other data.
Finding Text in an Image
------------------------
There is a python script written by Dan Vanderkam which locates text in an image and crops the image accordingly.  The [source code](https://github.com/danvk/oldnyc/blob/master/ocr/tess/crop_morphology.py, "Crop Morphology") is available on GitHub.
To grab only the latitude and longitude values, the code must be modified.  It is easy to start by cutting the text in half with this function:
```
def crop_half(img):
    height = img.size[1]
    height_half = height/2
    width = img.size[0]
    new = img.crop((0, height_half, width, height))
    
    return new
```
This is made possible by using PIL.
![alt Original Frame](frame.png)![alt Original Crop](cropped.png)![alt Cropped In Half](cropped_half.png)
The original, cropped, and cropped in half images.



