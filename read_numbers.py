#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

def read_numbers(img):

    img_train = cv2.imread('learner_done.png')
    img_test = cv2.imread(img)
    gray_train = cv2.cvtColor(img_train,cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 260 cells, each 16x24 size
    cells_train = [np.hsplit(row,10) for row in np.vsplit(gray_train,26)]
    # 9 cells, 16x24
    cells_test = [np.hsplit(gray_test,9)]

    # Make it into a Numpy array. It size will be (26,10,16,24)
    x_train = np.array(cells_train)
    x_test = np.array(cells_test)

    # Now we prepare train_data and test_data.

    train = x_train[:,:5].reshape(-1,384).astype(np.float32) # Size = (2500,400)
    #test = x[:,5:10].reshape(-1,384).astype(np.float32) # Size = (2500,400)
    test = x_test.reshape(-1,384).astype(np.float32)

    # Create labels for train and test data
    k = np.arange(13)
    train_labels = np.repeat(k,10)[:,np.newaxis]
    test_labels = train_labels.copy()

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train,train_labels)
    ret,result,neighbours,dist = knn.find_nearest(test,k=5)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    l = []
    for r in result:
        l.append(r)

    str1 = ''.join(str(e) for e in l)

    #print str1

    str2 = ""

    for e in str1:
        if(e != "[" and e != "]" and e != "." and e != " "):
	    str2+=str(e)

    str2 = str2.replace("11", "W", 1)
    str2 = str2.replace("10", "N", 1)
    str2 = str2.replace("12", ".", 1)


    return str2


print read_numbers('lat.png')
print read_numbers('long.png')
