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

