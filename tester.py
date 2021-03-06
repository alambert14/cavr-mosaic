#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw

def crop_alt(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((326,height-62,370,height-38))
    return new

def crop_time(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((150,height-62,278,height-38))
    return new

cap = cv2.VideoCapture('/home/cavr/160721_Videos/191000.MPG')
ret, img = cap.read()
cv2.imwrite('frame1.png',img)


frame = cv2.imread('frame1.png')    
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
pil_im = Image.fromarray(gray)
new = crop_time(pil_im)
new.save('crop_test.png')
cap.release()
