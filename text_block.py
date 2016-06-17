#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('frame.png')
edges = cv2.Canny(img,200,400)

cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
cv2.imshow( "Display window", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
