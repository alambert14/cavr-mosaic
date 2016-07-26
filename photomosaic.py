#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
from frame import Frame

mouse_x = 0
mouse_y = 0

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 50, qualityLevel = 0.5,
                       minDistance = 1,
                       blockSize = 2 )

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

def mod_correction(lon):
    if (lon[6:].find("N") != -1 or lon[6:].find("W") != -1):
        print "Value Error"
        return -1
    else:
        lon_dec = int(lon[6:])
        if lon_dec >= 400 and lon_dec <= 500:
            lon_dec = lon_dec-300
            first_half = lon[:6]
            lon = first_half + str(lon_dec)
        if lon_dec > 1000:
            lon_dec = str(1) + str(lon_dec)[2:]
            first_half = lon[:6]
            lon = first_half + str(lon_dec)
        return lon

def mod_correction2(lon):
    lon_dec = int(lon[6:])
    if lon_dec > 1000:
        lon_dec = str(1) + str(lon_dec)[2:]
        first_half = lon[:6]
        lon = first_half + str(lon_dec)
    
def click(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x = x
    mouse_y = y
    



#reads the numbers from the latitude and longitude images
def read_numbers(img):

    #loads the learner and testing file and converts them to grayscale
    img_train = cv2.imread('learner_done2.png')
    img_test = cv2.imread(img) #7,
                       #blockSize = 7 
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
    picture_list = []

    current_lat = "00000.000"
    current_lon = "00000.000"
    ind = 0

    #while there are still frames in the video
    while(True):
        if ind == 913:
            print "*********************** END OF TRIMMED VIDEO ***********************"
        ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        #cv2.waitKey(1)
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
        
        #print lon
        print lat,lon
    #processes the latitude and longitude values based on a few conditions:
        #they must both have 9 characters
        #they must contain a decimal point
        #they must begin with a certain string (based on Florida)
        #they must not vary very much from the frame before it
    #this is so bad data can be filtered out, such as when there is a splash
        #if the frame is accepted, it will be appended to a list of Frame objects

        ##### THIS WORKS ######
        if (len(lat) == 9 and len(lon) == 9):
            #print "Has 9 chars"
            if(lat.find('.') != -1 and lon.find('.') != -1):
                #print "Has a decimal point"
                if(lat.find('24N56') != -1 and lon.find('80W27') != -1):
                    lon = mod_correction(lon)
                    if lon == -1:
                        continue
                    #print lon
                    if current_lat == "00000.000" or current_lon == "00000.000":
                        current_lat = lat
                        current_lon = lon
                        #print "Current set"
                    #print "Is in florida"
                    lat_decimal = int(lat[6:])
                    lon_decimal = int(lon[6:])
                    current_lat_decimal = int(current_lat[6:])
                    current_lon_decimal = int(current_lon[6:])
                    #print str(ind)
                    #print "Is not first frame"
                    
                    #print "Current: "  + str(current_lat_decimal)
                    #print "New: " + str(lat_decimal)
                    if abs(current_lat_decimal - lat_decimal) <= 2 and abs(current_lon_decimal - lon_decimal) <= 2:
                        #print "Not much variation"
                        current_lat = lat
                        current_lon = lon
                        #print lat, lon
                        tile = Frame(crop_frame_new(pil_im),lat,lon)
                        frame_list.append(tile)
                        #cv2.imshow('frame', gray)
        picture_list.append(crop_frame_new(pil_im))
    
        ind=ind+1   
    print str(ind)
    print str(len(frame_list))
    #closes the stream and returns a list 
    cap.release()
    #cv2.waitKey(0)
    cv2.destroyAllWindows() 
    print "Frames processed...creating mosaic"
    return frame_list, picture_list

def first_stitch(frame, blank, max_lon, min_lat):
    #print "First Stitch"
    #TODO: better names
    #lat_dec = int(frame.lat[6:])

    lat_dec = 921
    lon_dec = 150
    #lon_dec = int(frame.lon[6:])
    #creates an index for the lat and lon values from 0-maximum
    print max_lon, min_lat
    lon_ind = max_lon-lon_dec
    lat_ind = lat_dec-min_lat
    #TODO: don't use numbers
    img_width = frame.size[1]
    img_height = frame.size[0]
    #determines the px position in the frame and pastes it to blank mosaic
    point = ((lon_ind*img_width),(lat_ind*img_height))
    print "Pasted at: " + str(point)
    blank.paste(frame, point)
    return point, blank

def stitch(blank, prev_img, this_img, point):
    #print "New Stitch"
    #TODO: Don't save and reopen, just convert to array
    prev_img.save('prev.png')
    this_img.save('new.png')
    prev_gray = cv2.imread('prev.png', cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.imread('new.png', cv2.IMREAD_GRAYSCALE)    

    #prev = np.array(prev_img)
    #prev_gray = prev.astype(np.uint8)
    cv2.imshow("img1", prev_gray)
    cv2.waitKey(1)
    #img = np.array(this_img)
    #img_gray = img.astype(np.uint8)
    #prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    #print "Finding good features to track"
    
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    #print str(point)
    for pt in p0:
        point_tup = (pt[0][0], pt[0][1])
        cv2.circle(prev_gray, point_tup, 5, (255,0,0), -1)
    if p0 == None:
        cv2.imshow("image",prev_gray)
        cv2.setMouseCallback("image", click)
        cv2.waitKey(0)
        p0 = [[mouse_x, mouse_y]]
    #print p0
    #calculate the optical flow
    #print "Calculating optical flow"
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, img_gray, p0, None, **lk_params)

    good_new = p1[st==1] #Only takes values that have a status of 1 (means that optical flow has been found)
    good_old = p0[st==1]
    good_err = err[st==1]
    #error is from 0-10
    lowest_err = 10 
    i = 0
    lowest_err_ind = 0
    #find the lowest amount of error
    #print "Finding lowest error"
    while i < len(good_err):
        if good_err[i] < lowest_err:
            lowest_err = good_err[i]
            lowest_err_ind = i
        i += 1
    
    #print "Pasting the image"
    #get the points from the feature with lowest error
    a,b = good_new[lowest_err_ind]
    c,d = good_old[lowest_err_ind]
    
    #determine the displacement between the two features
    hor_disp = a-c
    vert_disp = b-d
    #create a point to paste the new image based on this displacement
    new_point = ((int(point[0]+hor_disp)), (int(point[1]+vert_disp)))
    new_img = Image.fromarray(img_gray)
    print "Pasted at: " + str(new_point)
    blank.paste(new_img, new_point)
    blank.save('inprogress.jpg')
    return new_point, blank

#creates a mosaic image from the list of Frames
def mosaic(frame_list_rough, picture_list):
    print "Beginning to stitch mosaic...please wait..."
    frame_list = []
    #print frame_list
    lat_dec_list = []
    lon_dec_list = []
    
    prev_lat = 0
    prev_lon = 0
    ind = 0
    #for each frame, create a list of the longitude and latitude decimals
    for frame in frame_list_rough:
        #test to make sure there are no letters in the decimal
        #if (frame.lon[6:].find("N") != -1 or frame.lon[6:].find("W") != -1 or frame.lat[6:].find("N") != 01 or frame.lat[6:].find("W") != -1):
        #    print "Value Error"
        #    continue
        #else:
        lat_dec = frame.lat[6:]
        lon_dec = frame.lon[6:]
        #test to make sure the numbers are consecutive
        #if ind == 0:
        #    curr_lat = lat_dec
        #    curr_lon = lon_dec
        #        
        #    frame_list.append(frame)
        #    lat_dec_list.append(int(lat_dec))
        #    lon_dec_list.append(int(lon_dec))
        #    print "Finished first iteration"
            
        #else:
        #    #print "Checking consecutivity"
        #    if int(curr_lat) - int(lat_dec) >= -1:
        #        #print "Lat1 succeeded"
        #        if int(curr_lat) - int(lat_dec) <= 1:
        #            #print "Lat2 succeeded"
        #            if int(curr_lon) - int(lon_dec) >= -1:
        #                #print "Lon1 succeeded"
        #                if int(curr_lon) - int(lon_dec) >= 1:
        #                    #print "Lon2 succeeded"
        #                    curr_lat = lat_dec
        #                    curr_lon = lon_dec
        #            
        #                    frame_list.append(frame)
        lat_dec_list.append(int(lat_dec))
        lon_dec_list.append(int(lon_dec))
        frame_list.append(frame)
        #ind += 1

    #print str(len(lat_dec_list))
    #print str(len(frame_list))
    print str(lat_dec_list)
    print str(lon_dec_list)

    #recognize the largest and smallest latitude and longitude values
    #print "Finding maximum and minimum"
    maximum_lat = max(lat_dec_list)
    minimum_lat = min(lat_dec_list) - 1
    maximum_lon = max(lon_dec_list)
    minimum_lon = min(lon_dec_list)

    #finds the range of the latitude and longitude values
    #TODO: find a range function to use instead
    #print "Calculating range"317

    lat_range = maximum_lat - minimum_lat
    lon_range = maximum_lon - minimum_lon

    #finds the dimensions of the image so the size of the mosaic can be calculated
    #print "Finding size of the mosaic"
    img_width = frame_list[0].img.size[0]
    img_height = frame_list[0].img.size[1]
    size = (((lon_range+1)*img_width),((lat_range+1)*img_height))

    #create a new blank image for the mosaic
    #print "Creating a new blank image"
    blank = Image.new('RGB',size,(255,153,153))
    #print "Creating a new point"
    point = (0,0)

    #for each frame, stitch it to the one previous
    i = 0
    #print "About to loop 1"

    
    while i < len(picture_list):
        if i == 0:
            #print "About to do first stich"
            point, blank = first_stitch(picture_list[i], blank, maximum_lon, minimum_lat)
            #blank.save('mosaic1.jpg')
        else:
            print "About to do next stitch"
            point, blank = stitch(blank, picture_list[i-1], picture_list[i], point) #TODO: more params
            #blank.save('mosaic1.jpg')
           
        i += 1

    print "Saving Mosaic to mosaic1.png"
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
    
    
    frames, pictures = text_capture('/home/cavr/160721_Videos/191000.MPG')
    mosaic(frames, pictures)










