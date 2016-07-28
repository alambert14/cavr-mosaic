#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw

def crop_frame(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((2,0,width-30,height-60))
    return new
    #688x420


if __name__ == "__main__":
    #for fn in os.listdir('/home/cavr/Aaron'):
    #    fn.open()
    #    frame = cv2.imread(fn.name)
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    pil_im = Image.fromarray(gray)
    #    new = crop_frame(pil_im)
    #    new.save(fn.name)
    cap = cv2.VideoCapture('/home/cavr/160721_Videos/191000.MPG')
    ind = 0
    while(True):
        ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        #cv2.waitKey(1)
        if frame is None:
            break
        pil = Image.fromarray(frame)
        pil.save("/home/cavr/Test_images/" + str(ind) + ".jpg")
        ind+=1
