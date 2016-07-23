#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from frame import Frame

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#1: get video file (rosparam) later, easy
#2: read video into OpenCV
    # capture
    # use http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html#lucas-kanade
    
#3: get lat/lon text recognition data into GLOBAL lists using Abby's thing
#4: crop lat/lon text out (maybe right hand side also with weird edges?)
#5: use abby's code to get size of output mosaic
#6: use CV code for optical flow...for track good features, corner detection? 
    #also...figure this out
    #also...later devel, able to merge perc_fls_sim + this code where, if a desired feature to track is clicked, this tracks it, outputs mosaic with a DIFFERENT COLOR BROOOOO
#7: output mosaic



### Functions called ###
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
def crop_frame_old(img):
    height = img.size[1]
    width = img.size[0]
    new = img.crop((0,0,width-15,height))
    return new
    #new width is 705px



#crops the black strip and the lat lon labels out of the frame
def crop_frame_new(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((2,0,width-30,height-60))
    return new
    #688x420



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
def text_capture(video_file):

    cap = cv2.VideoCapture(video_file)
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
        print lat,lon
    #processes the latitude and longitude values based on a few conditions:
        #they must both have 9 characters
        #they must contain a decimal point
        #they must begin with a certain string (based on Florida)
        #they must not vary very much from the frame before it
    #this is so bad data can be filtered out, such as when there is a splash
        #if the frame is accepted, it will be appended to a list of Frame objects
        if (len(lat) == 9 and len(lon) == 9):
            #print "Has 9 chars"
            if(lat.find('.') != -1 and lon.find('.') != -1):
                #print "Has a decimal point"
                if(lat.find('24N56') != -1 and lon.find('80W27') != -1):
                    #print "Is in florida"
                    lat_decimal = int(lat[6:])
                    lon_decimal = int(lon[6:])
                    current_lat_decimal = int(current_lat[6:])
                    current_lon_decimal = int(current_lon[6:])
                    if(ind != 0):
                        #print "Is not first frame"
                        if((current_lat_decimal - lat_decimal >= -2 and current_lat_decimal - lat_decimal <= 2) and (current_lon_decimal - lon_decimal >= -2 and current_lon_decimal - lon_decimal <= 2)):
                            #print "Not much variation"
                            current_lat = lat
                            current_lon = lon
                            #print lat, lon
                            tile = Frame(crop_frame_old(pil_im),lat,lon)
                            frame_list.append(tile)
                            #cv2.imshow('frame', gray)
                        else:
                            current_lat = lat
                            current_lon = lon
                    else:
                        current_lat = lat
                        current_lon = lon
                        tile = Frame(crop_frame_old(pil_im),lat,lon)
                        frame_list.append(tile)
                        cv2.imshow('frame',gray)
    
        ind=ind+1   

    #closes the stream and returns a list 
    cap.release()
    #cv2.waitKey(0)
    cv2.destroyAllWindows() 
    print "Frames processed...creating mosaic"
    return frame_list

def first_stitch(frame, blank, max_lon, min_lat):
    print "First Stitch"
    #TODO: better names
    lat_dec = int(frame.lat[6:])
    lon_dec = int(frame.lon[6:])
    #creates an index for the lat and lon values from 0-maximum
    lon_ind = maximum_lon-lon_dec
    lat_ind = lat_dec-minimum_lat
    #TODO: don't use numbers
    img_width = frame.img.size[1]
    img_height = frame.img.size[0]
    #determines the px position in the frame and pastes it to blank mosaic
    point = ((lon_ind*img_width),(lat_ind*img_height))
    blank.paste(point, frame.img)
    return point, blank

def stitch(blank, prev_img, this_img, point):
    print "New Stitch"
    prev = np.array(prev_img)
    img = np.array(this_img)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    print "Finding good features to track"
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #calculate the optical flow
    print "Calculating optical flow"
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, img_gray, p0, None, **lk_params)

    good_new = p1[st==1] #Only takes values that have a status of 1 (means that optical flow has been found)
    good_old = p0[st==1]
    good_err = err[st==1]
    #error is from 0-10
    lowest_err = 10 
    i = 0
    lowest_err_ind = 0
    #find the lowest amount of error
    print "Finding lowest error"
    while i < len(good_err):
        if good_err[i] < lowest_err:
            lowest_err = good_err[i]
            lowest_err_ind = i
        i += 1
    
    print "Pasting the image"
    #get the points from the feature with lowest error
    a,b = good_new[lowest_err_ind]
    c,d = good_old[lowest_err_ind]
    
    #determine the displacement between the two features
    hor_disp = a-c
    vert_disp = b-d
    #create a point to paste the new image based on this displacement
    new_point = ((point[0]+hor_disp),(point[1]+vert_disp))
    new_img = Image.fromarray(img_gray)
    blank.paste(new_point, new_img)
    return point, blank
    


    #find the best feature, find the corresponding feature in the prev. image, and then paste relative to the prev image

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

#creates a mosaic image from the list of Frames
def mosaic(frame_list_rough):
    frame_list = []
    lat_dec_list = []
    lon_dec_list = []
    
    prev_lat = 0
    prev_lon = 0
    ind = 0
    #for each frame, create a list of the longitude and latitude decimals
    for frame in frame_list:
        #test to make sure there are no letters in the decimal
        try:
            lat_dec = frame.lat[6:]
            lon_dec = frame.lon[6:]
        except ValueError:
            print "Value Error"
            continue
        #test to make sure the numbers are consecutive
        if ind == 0:
            curr_lat = lat_dec
            curr_lon = lon_dec
                
            frame_list.append(frame)
            lat_dec_list.append(int(lat_dec))
            lon_dec_list.append(int(lon_dec))
            print "Finished first iteration"
            
        else:
            print "Checking consecutivity"
            if curr_lat - lat_dec >= -1:
                print "Lat1 succeeded"
                if curr_lat - lat_dec <= 1:
                    print "Lat2 succeeded"
                    if curr_lon - lon_dec >= -1:
                        print "Lon1 succeeded"
                        if curr_lon - lon_dec >= 1:
                            print "Lon2 succeeded"
                            curr_lat = lat_dec
                            curr_lon = lon_dec
                    
                            frame_list.append(frame)
                            lat_dec_list.append(int(lat_dec))
                            lon_dec_list.append(int(lon_dec))
        ind += 1

    print str(lat_dec_list)

    #recognize the largest and smallest latitude and longitude values
    print "Finding maximum and minimum"
    maximum_lat = max(lat_dec_list)
    minimum_lat = min(lat_dec_list)
    maximum_lon = max(lon_dec_list)
    minimum_lon = min(lon_dec_list)

    #finds the range of the latitude and longitude values
    #TODO: find a range function to use instead
    print "Calculating range"
    lat_range = maximum_lat - minimum_lat
    lon_range = maximum_lon - minimum_lon

    #finds the dimensions of the image so the size of the mosaic can be calculated
    print "Finding size of the mosaic"
    img_width = frame_list[0].img.size[0]
    img_height = frame_list[0].img.size[1]
    size = ((lon_range+1)*img_width,(lat_range+1)*img_height)

    #create a new blank image for the mosaic
    print "Creating a new blank image"
    blank = Image.new('RGB',size,(255,153,204))
    print "Creating a new point"
    point = (0,0)

    #for each frame, stitch it to the one previous
    i = 0
    print "About to loop 1"
    while i < len(frame_list):
        if i == 0:
            print "About to do first stich"
            point, blank = first_stitch(frame_list[i], blank, maximum_lon, minimum_lat)
        else:
            print "About to do next stitch"
            point, blank = stitch(frames_list[i-1], frames_list[i], point) #TODO: more params
           
        i += 1

    blank.save('mosaic1.jpg')

    

    
    ##creates a list of long_range+1 lists filled with lat_range+1 NoneTypes
    #frame_array = [[] for i in xrange(lon_range+1)]
    #ind = 0
    #while(ind < len(frame_array)):
    #    frame_array[ind] = [None] * (lat_range+1) 
    #    ind = ind+1


    #adds each Frame object to its spot in a matrix
    #for frame in frame_list:
    #    lat_dec = int(frame.lat[6:])
    #    lon_dec = int(frame.lon[6:])
    #    #creates an index for the lat and lon values from 0-maximum
    #    frame_array[(maximum_lon-lon_dec)][(lat_dec-minimum_lat)] = frame      
    
    #iterates through the 2d list, placing each frame where it belongs on the blank
    #i = 0
    #j = 0
    #while(i < len(frame_array)):
    #    j = 0
    #    while(j < len(frame_array[i])):
    #        place = ((i*img_width),(j*img_height))
    #        if frame_array[i][j] is not None:
    #            img = frame_array[i][j].img
    #            blank.paste(img,place)
    #        j = j+1
    #    i = i+1
    ##saves the mosaic
    #blank.save('mosaic1.png')
    #print "Mosaic saved at mosaic.png"





### Main loop, calls above functions ###
if __name__ == '__main__':
    #rospy.init_node('photomosaic')
    #rospack = rospkg.RosPack()
    #
    ## configuration, get video file (for later)
    #fbn_raw_topic = rospy.get_param('/fbn_raw_topic','/fbn_raw')
    #fbn_viz_topic = rospy.get_param('~fbn_viz_topic','/fbn_viz')
    #f_opencv_topic = rospy.get_param('~f_opencv_topic','/fls_img')
    #img_topic = rospy.get_param('~sim_fls_com','/son_img_com')
    #prob_detect = rospy.get_param('~prob_detect',0.5)
    #meas_noise_stddev = rospy.get_param('~meas_noise_stddev',0.5)
    #detect_delay = rospy.get_param('~detect_delay',0.2)
    #sonar_hor_fov = rospy.get_param('~sonar_hor_fov',45.0)
    #sonar_vert_fov = rospy.get_param('~sonar_vert_fov',18.0)
    #sonar_range = rospy.get_param('~sonar_range',50.0)
    #rate = rospy.get_param('~rate',1.0)
    #ref_frame_id = rospy.get_param('/ref_frame','world_ned')
    ##NOTE: sensor frame in simulation should be fls_hor_true this ensures 
    ##detections are calculated based on true positions
    #sensor_frame_id = rospy.get_param('~sensor_frame_id','fls_hor')
    #video_file = rospy.get_param('~video_file','/etc/test.MPG')
    #path = rospack.get_path('photomosaic')
    # 
    ##Calls text_capture function above given rosparam input of specific video_file
    ##TODO:stores global lists of lat/lon
    ##Stores returned frame_list here
    #frame_list = text_capture(video_file))
    
    mosaic(text_capture('/home/cavr/160721_Videos/191000_trim.MPG'))











