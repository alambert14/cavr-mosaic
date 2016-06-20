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

**TODO:** Determine whether or not it is important to use his script.  The text can just be cropped using manual values as it is in later steps.  However, analyzing his code was informative in my understanding of vision processing. 

![alt Original Frame](frame.png)![alt Original Crop](cropped.png)![alt Cropped In Half](cropped_half.png)

The original, cropped, and cropped in half images.
This can be cropped further to grab both the latitude a nd longitude texts with padding on the sides of 1px and padding on top and bottom of 2px.  This is required to run the text recognition software.
Putting It Together: Grabbing the Latitude and Longitude Values
---------------------------------------------------------------
Once the frame is cropped (one cropped image for latitude and one for longitute), they can be fed through the text recognition software.  I modified the digit recognition script written by the OpenCV developers to read the test image from a file.  The results of the test are not a single string, so I converted it to a string and did string manipulation to produce a string that looks as such: 24N56.898

A script can be created to run both the cropping script and text recognition script one after the other.  This will read a picture and output the latitute and longitute values:
```
./crop.py frame.png &
sleep 2
./read_numbers.py &
sleep 2
rm lat.png &
sleep 1
rm long.png &
```
Goal 3: Using Latitude and Longitute to Create A Mosaic
=======================================================
Using the Correct Images
------------------------
There seems to be relatively enough varience in the latitude and longitude information for a photo mosaic to be formed.  Right now, the video responds with steady data until something gets in the way, such as a fin or a splash.
It is necessary to filter out all of the "bad" or repetative longitute and latitude values using a series of if statements.  So far, these statements make the assumption that the video is being captured in Florida.
The images that have correct lat and long values are stored in Frame objects, which are compiled into a list that is  passed to the function that will piece together the mosaic.
Making the Mosaic
-----------------
Since this code assumes the robot is in Florida, the code only uses the decimals after the amount of minutes, since it it is only moving a thousanth of a minute every few seconds (i.e. 24N56.898 would be assumed as 898).  I used the maximum and minimum lat and long values from the video to create a base for the mosaic that will eventually be filled with image tiles.  I created a 19x16 list that will store all of the Frame objects.  For the latitude values, I subtracted the minimum latitude value from the actual value to give it a latitude index.
**Note:** Since Florida is in the Western Hemisphere, all of the longitude values are assumed to be negative, and an ajustment has to be made (subtracting the value from the maximum long value to get its longitude index).
Once they have these indeces, the pictures can be pasted into a big picture file that will be the mosaic.  The list of lists of Frame objects is iterated through and each occupied space is pasted into the mosaic.
Issues
------
There are not as many images as I would have hoped, and this is due to not enough varience in the latitude and longitude information.

This is what the mosaic looks like currently (from the video file I have):

![alt Mosaic](mosaic.png)
