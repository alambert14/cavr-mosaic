#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from frame import Frame

#crops the frame to get the latitude and longitude
def crop(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((22,height-32,342,height-8))
    return new

#crops the frame to only get the latitude
def crop_lat(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((0,0,144,height))
    return new 

#crops the frame to only get the longitude
def crop_lon(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((176,0,320,height))
    return new

#crops the black strip out of the frame
def crop_frame(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((0,0,width-15,height))
    return new
    #new width is 705px

#reads the numbers from the latitude and longitude images
def read_numbers(img):

    #loads the learner and testing file and converts them to grayscale
    img_train = cv2.imread('learner_done2.png')
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

    #combines the matched numbers into a string, outputs a list
    str1 = ''.join(str(e) for e in l)

    str2 = ""

    #deletes all of the periods in the string
    for e in str1:
	if(e != "."):
	    str2+=str(e)

    #replaces with the north and west markers and the periods
    str2 = str2.replace("11", "W", 1)
    str2 = str2.replace("10", "N", 1)
    str2 = str2.replace("12", ".", 1)

    #removes the rest of the list string elements that aren't numbers
    str3 = ""
    for e in str2:
        if(e != "[" and e != "]" and e != " "):
	    str3+=str(e)

    #returns the processed string
    return str3
    
#reads the latitude and longitude values off of each frame of a video and stores them in a list
def text_capture():

    cap = cv2.VideoCapture('/home/cavr/160721_Videos/191000.MPG')
    print "Reading and processing frames may take a minute"
    print "Please Wait..."
    frame_list = []

    current_lat = "00000.000"
    current_lon = "00000.000"
    ind = 0

    #while there are still frames in the video
    while(True):

        ret, frame = cap.read()
	if frame is None:
	    break

 	#converts the frame to grayscale and then to a PIL image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_im = Image.fromarray(gray)
    
        #crops the frame to latitude and longitude
        first_crop = crop(pil_im)
        lat_im = crop_lat(first_crop)
        lon_im = crop_lon(first_crop)

	#saves the latitude and longitude images
        lat_im.save('lat.png')
        lon_im.save('lon.png') 
    
        #reads the numbers from each frame
        lat = read_numbers('lat.png')
        lon = read_numbers('lon.png')
	#processes the latitude and longitude values based on a few conditions:
	    #they must both have 9 characters
	    #they must contain a decimal point
	    #they must begin with a certain string (based on Florida)
	    #they must be different from the previous frame (so they are all unique)
	    #they must not vary very much from the frame before it
	#this is so bad data can be filtered out, such as when there is a splash
        #if the frame is accepted, it will be appended to a list of Frame objects
        if (len(lat) == 9 and len(lon) == 9):
	    if(lat.find('.') != -1 and lon.find('.') != -1):
	        if(lat.find('24N56') != -1 and lon.find('80W27') != -1):
		    lat_decimal = int(lat[6:])
    		    lon_decimal = int(lon[6:])
  
       		    current_lat_decimal = int(current_lat[6:])
 		    current_lon_decimal = int(current_lon[6:])
		    if(lat_decimal != current_lat_decimal or lon_decimal != current_lon_decimal):
		        if(ind != 0):

		            if((current_lat_decimal - lat_decimal >= -2 and current_lat_decimal - lat_decimal <= 2) and (current_lon_decimal - lon_decimal >= -2 and current_lon_decimal - lon_decimal <= 2)):
			        current_lat = lat
			        current_lon = lon
			        tile = Frame(crop_frame(pil_im),lat,lon)
			        frame_list.append(tile)
			        cv2.imshow('frame', gray)
		        else:
			    current_lat = lat
			    current_lon = lon
			    tile = Frame(crop_frame(pil_im),lat,lon)
			    frame_list.append(tile)
			    cv2.imshow('frame',gray)

        ind=ind+1   

    #closes the stream and returns a list 
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    print "Frames processed...creating mosaic"
    return frame_list

#creates a mosaic image from the list of Frames
def mosaic(frame_list):
    lat_dec_list = []
    lon_dec_list = []

    #for each frame, create a list of the longitude and latitude decimals
    for frame in frame_list:
	lat_dec = frame.lat[6:]
	lon_dec = frame.lon[6:]
	lat_dec_list.append(int(lat_dec))
	lon_dec_list.append(int(lon_dec))

    #recognize the largest and smallest latitude and longitude values
    maximum_lat = max(lat_dec_list)
    minimum_lat = min(lat_dec_list)
    maximum_lon = max(lon_dec_list)
    minimum_lon = min(lon_dec_list)

    #finds the range of the latitude and longitude values
    #TODO: find a range function to use instead
    lat_range = maximum_lat - minimum_lat
    lon_range = maximum_lon - minimum_lon

    #finds the dimentions of the image so the size of the mosaic can be calculated
    img_width = frame_list[0].img.size[0]
    img_height = frame_list[0].img.size[1]
    size = ((lon_range+1)*img_width,(lat_range+1)*img_height)

    #create a new blank image for the mosaic
    blank = Image.new('RGB',size,(255,153,204))

    
    #creates a list of long_range+1 lists filled with lat_range+1 NoneTypes
    frame_array = [[] for i in xrange(lon_range+1)]
    ind = 0
    while(ind < len(frame_array)):
	frame_array[ind] = [None] * (lat_range+1) 
	ind = ind+1


    #adds each Frame object to its spot in a matrix
    for frame in frame_list:
	lat_dec = int(frame.lat[6:])
	lon_dec = int(frame.lon[6:])
	#creates an index for the lat and lon values from 0-maximum
	frame_array[(maximum_lon-lon_dec)][(lat_dec-minimum_lat)] = frame
		
    
    #iterates through the 2d list, placing each frame where it belongs on the blank
    i = 0
    j = 0
    while(i < len(frame_array)):
	j = 0
	while(j < len(frame_array[i])):
	    place = ((i*img_width),(j*img_height))
	    if frame_array[i][j] is not None:
	        img = frame_array[i][j].img
	        blank.paste(img,place)
	    j = j+1
	i = i+1
    #saves the mosaic	
    blank.save('mosaic1.png')
    print "Mosaic saved at mosaic.png"

if __name__ == '__main__':
    mosaic(text_capture())


