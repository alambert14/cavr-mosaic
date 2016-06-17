#! /usr/bin/env python

import cv2
import numpy as np

video = cv2.VideoCapture("../140723_162428.MPG")

#num_frames = video.get(CV_CAP_PROP_FRAME_COUNT)

#while(video.isOpened()):
ret, frame = video.read()
cv2.imwrite('frame.png',frame)
cv2.imshow('image',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

