#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from frame import Frame

def crop(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((22,height-32,342,height-8))
    return new

def crop_lat(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((0,0,144,height))
    return new 

def crop_lon(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((176,0,320,height))
    return new

def read_numbers(img):
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

    str1 = ''.join(str(e) for e in l)

    #print str1

    str2 = ""

    for e in str1:
	if(e != "."):
	    str2+=str(e)

    str2 = str2.replace("11", "W", 1)
    str2 = str2.replace("10", "N", 1)
    str2 = str2.replace("12", ".", 1)

    str3 = ""
    for e in str2:
        if(e != "[" and e != "]" and e != " "):
	    str3+=str(e)

    return str3
    
def text_capture():

    cap = cv2.VideoCapture('../140723_163000.MPG')

    frame_list = []

    current_lat = "11111.111"
    current_lon = "11111.111"
    ind = 0

    while(True):

        ret, frame = cap.read()
	if frame is None:
	    break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pil_im = Image.fromarray(gray)
    

        first_crop = crop(pil_im)
        lat_im = crop_lat(first_crop)
        lon_im = crop_lon(first_crop)

        lat_im.save('lat.png')
        lon_im.save('lon.png') 
    
        #time.sleep(5)

        lat = read_numbers('lat.png')
        lon = read_numbers('lon.png')
        #print lat,lon

    

        if (len(lat) == 9 and len(lon) == 9):
	    if(lat.find('.') != -1 and lon.find('.') != -1):
	        if(lat.find('24N56') != -1 and lon.find('80W27') != -1):
		    lat_decimal = int(lat[6:])
    		    lon_decimal = int(lon[6:])
  
       		    current_lat_decimal = int(current_lat[6:])
 		    current_lon_decimal = int(current_lon[6:])
		    if(lat_decimal != current_lat_decimal or lon_decimal != current_lon_decimal):
		        print current_lat_decimal,current_lon_decimal
		        if(ind != 0):

		            if((current_lat_decimal - lat_decimal >= -2 and current_lat_decimal - lat_decimal <= 2) and (current_lon_decimal - lon_decimal >= -2 and current_lon_decimal - lon_decimal <= 2)):
			        current_lat = lat
			        current_lon = lon
			        tile = Frame(gray,lat,lon)
			        frame_list.append(tile)
			        #print lat_decimal,lon_decimal
			        cv2.imshow('frame', gray)
		        else:
			    current_lat = lat
			    current_lon = lon
			    tile = Frame(gray,lat,lon)
			    frame_list.append(tile)
			    #print lat_decimal, lon_decimal
			    cv2.imshow('frame',gray)

    

        
        ind=ind+1

    

    cap.release()
    cv2.destroyAllWindows()
    return frame_list

def mosaic(frame_list):
    
    lat_dec_list = []
    lon_dec_list = []

    for frame in frame_list:
	lat_dec = frame.lat[6:]
	lon_dec = frame.lon[6:]
	lat_dec_list.append(int(lat_dec))
	lon_dec_list.append(int(lon_dec))
    print lat_dec_list
    print lon_dec_list

    maximum_lat = max(lat_dec_list)
    minimum_lat = min(lat_dec_list)
    maximum_lon = max(lon_dec_list)
    minimum_lon = min(lon_dec_list)

    lat_range = maximum_lat - minimum_lat
    lon_range = maximum_lon - minimum_lon

    size = (lon_range*720,lat_range*480)

    blank = Image.new('RGB',size,(0,0,0))


    shape = (lon_range,lat_range)

    #frame_array = np.empty(shape,PyObject,'C')
    frame_array = [[] for i in xrange(lon_range+1)]

    ind = 0
    while(ind < len(frame_array)):
	frame_array[ind] = [None, None, None, None, None, None]
	ind = ind+1

    #for l in frame_array:
#	l = [None] * 6
#	l.append(3) 
	#l = None

    print frame_array

    for frame in frame_list:
	lat_dec = int(frame.lat[6:])
	lon_dec = int(frame.lon[6:])
	print "Lat",lat_dec,minimum_lat
	print "Lon",maximum_lon,lon_dec,minimum_lon
	#print frame_array[(maximum_lon-lon_dec)]
	#print frame_array[0][(lat_dec-minimum_lat)]
	#frame_array[(lat_dec-minimum_lat)][(maximum_lon-lon_dec)] = frame
	frame_array[(maximum_lon-lon_dec)][(lat_dec-minimum_lat)] = frame
	print "Frame allocated to position " + str(maximum_lon-lon_dec)+" "+str(lat_dec-minimum_lat)
	

    print "length of frame_array: " + str(len(frame_array))
    print "length of frame_array[0]: " + str(len(frame_array[0]))
    i = 0
    j = 0
    while(i < len(frame_array)):
	j = 0
	while(j < len(frame_array[i])):
	    place = ((i*720),(j*480))
	    if frame_array[i][j] is not None:
	        img = Image.fromarray(frame_array[i][j].img)
	        blank.paste(img,place)
		print "Image at position " + str(i) + " " + str(j) + " pasted!"
	    else:
		print "There is no image here!"
	    j = j+1
	    print "looping j again!"
	i = i+1
	print "looping i again!"
	

    blank.save('mosaic.png')

if __name__ == '__main__':
    mosaic(text_capture())


