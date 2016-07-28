#! /usr/bin/env python

import cv2
import numpy as np
from PIL import Image, ImageDraw
import piexif
import time

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

def crop_alt(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((326,height-62,374,height-38))
    return new

#crops the frame to only get the time
def crop_time(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((150,height-62,278,height-38))
    return new

#crops the black strip and the lat lon labels out of the frame
def crop_frame(img):
    width = img.size[0]
    height = img.size[1]
    new = img.crop((2,0,width-30,height-60))
    return new
    #688x420

def mod_correction(lon):
    if (lon[6:].find("N") != -1 or lon[6:].find("W") != -1 or lon[6:].find(".") != -1):
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



#Sets the GPS metadata for the image
def setGPS(lat, lon, i, img):
    img.save('/home/cavr/GPSImages' + str(i) + '.jpg')
    exifimg = Image.open('/home/cavr/GPSImages' + str(i) + '.jpg')
    ind = lat.find("N")
    lat_min = lat[ind+1:]
    lat_min2 = ""
    for e in lat_min:
        if(e != "."):
            lat_min2+=str(e)

    lat_min = int(lat_min2)

    ind = lon.find("W")
    lon_min = lon[ind+1:]
    lon_min2 = ""
    for e in lon_min:
        if(e != "."):
            lon_min2+=str(e)

    lon_min = int(lon_min2)
    
    #print exifimg.info
    #exif_dict = piexif.load(exifimg.info["exif"])
    
    exif_dict = {}
    exif_dict["GPS"] = {}
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = ((24, 1), (lat_min, 1000), (0, 1))
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N'

    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = ((80, 1), (lon_min, 1000), (0, 1))
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'W'

    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (70,10)
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0

    exif_bytes = piexif.dump(exif_dict)
    exifimg.save('/home/cavr/GPSImages/' + str(i) + '.jpg', "jpeg", exif=exif_bytes)

#reads the state data from a frame
def read_numbers(img, state):

    #loads the learner and testing file and converts them to grayscale
    img_train = cv2.imread('learner_done3.png')
    img_test = cv2.imread(img) #7,
                       #blockSize = 7 
    gray_train = cv2.cvtColor(img_train,cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 260 cells, each 16x24 size
    cells_train = [np.hsplit(row,10) for row in np.vsplit(gray_train,28)]
    # 9 cells, 16x24
    #cells_test = [np.hsplit(gray_test,9)]
    #if the state data is latlon
    if state == 0:
        cells_test = [np.hsplit(gray_test,9)]
    #if the state data is altitude    
    elif state == 1:
        cells_test = [np.hsplit(gray_test,3)]
    #if the state data is time
    elif state == 2:
        cells_test = [np.hsplit(gray_test,8)]
    # Make it into a Numpy array. It size will be (26,10,16,24)
    x_train = np.array(cells_train)
    x_test = np.array(cells_test)

    # Now we prepare train_data and test_data.

    train = x_train[:,:10].reshape(-1,384).astype(np.float32) # Size = (2500,400)
    #test = x[:,5:10].reshape(-1,384).astype(np.float32) # Size = (2500,400)
    test = x_test.reshape(-1,384).astype(np.float32)

    # Create labels for train and test data
    k = np.arange(14)
    train_labels = np.repeat(k,20)[:,np.newaxis]
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
    str2 = str2.replace("13", ":", 2)

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
    #frame_list = []
    #picture_list = []

    current_lat = "00000.000"
    current_lon = "00000.000"
    #if for video 191000.MPG
    curr_uniq_lat = "24N56.921"
    curr_uniq_lon = "80W27.150"
    ind = 0
    est_time = (10*60)+1
    prev_time = est_time
    del_lat = 0
    del_lon = 0
    est_time1 = est_time
    first_time = True
    

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
        alt_im = crop_alt(pil_im)
        time_im = crop_time(pil_im)
        lat_im = crop_lat(first_crop)
        lon_im = crop_lon(first_crop)

        #saves the latitude and longitude images
        lat_im.save('lat.png')
        lon_im.save('lon.png') 
        alt_im.save('alt.png')
        time_im.save('time.png')
        #reads the numbers from each frame
        lat = read_numbers('lat.png', 0)
        lon = read_numbers('lon.png', 0)
        
        alt = read_numbers('alt.png', 1)
        tm = read_numbers('time.png', 2)
        
        cropped_img = crop_frame(pil_im)
        final_img = cropped_img
        #print lon
        #print lat,lon
        #print alt
        #print time

        #processes the time data based on a few conditions:
            #it must be 8 characters
            #it must contain 2 colons
            #it must begin with 19
        #this is so bad data can be filtered out
        #if the data is not good, it will be determined by interpolating a function
        good_time = False
        if ind % 30 == 0:
            est_time = est_time+1
        if len(tm) == 8:
            colon_index = tm.find(":") 
            if (colon_index != -1 and tm[colon_index+1:].find(":") != -1):
                if tm[:2].find("19") != -1:
                    current_time = (int(tm[3:5])*10) + int(tm[6:])
                    if abs(est_time - current_time) >= 2:
                        if(current_time != prev_time):    
                            good_time = True
                        
                    
                     
                    
                


        #processes the latitude and longitude values based on a few conditions:
            #they must both have 9 characters
            #they must contain a decimal point
            #they must begin with a certain string (based on Florida)
            #they must not vary very much from the frame before it
        #this is so bad data can be filtered out, such as when there is a splash
        #if the frame is accepted, it will be appended to a list of Frame objects
        good_data = False
        
        #### This is the text recognition portion of this script. ####
        # It checks for char length of 9 for lat and lon which weeds out bad data that occurs
        if (len(lat) == 9 and len(lon) == 9):
            if(lat.find('.') != -1 and lon.find('.') != -1):
                if lat.find('24N56') != -1 and lon.find('80W27') != -1:
                    lon = mod_correction(lon)
                    if lon != -1:
                        if current_lat == "00000.000" or current_lon == "00000.000":
                            current_lat = lat
                            current_lon = lon
                        lat_decimal = int(lat[6:])
                        lon_decimal = int(lon[6:]) 
                        current_lat_decimal = int(current_lat[6:])
                        current_lon_decimal = int(current_lon[6:])
                        
                            
                        if abs(current_lat_decimal - lat_decimal) <= 5 and abs(current_lon_decimal - lon_decimal) <= 5:
                        
                            current_lat = lat
                            current_lon = lon
                            print lat, lon
                            good_data = True
                            
                            
                            #tile = Frame(crop_frame(pil_im),lat,lon)
                            #frame_list.append(tile)
                            #cv2.imshow('frame', gray)
                            if(lat_decimal != current_lat_decimal or lon_decimal != current_lon_decimal):
                                if not first_time:
                                    est_time1 = est_time2
                                curr_uniq_lat_dec = int(curr_uniq_lat[6:])
                                curr_uniq_lon_dec = int(curr_uniq_lon[6:])
                                uniq_lat_dec = int(lat[6:])
                                uniq_lon_dec = int(lon[6:])
                                del_lat = uniq_lat_dec - curr_uniq_lat_dec
                                del_lon = uniq_lon_dec - curr_uniq_lon_dec
                                est_time2 = est_time
                                first_time = False
                                

        if good_time and del_lat != 0 and del_lon != 0:
            calc_lat = (del_lat/est_time2-est_time1)*(est_time2-est_time)
            calc_lon = (del_lon/est_time2-est_time1)*(est_time2-est_time)
            tile = crop_frame(pil_im)

            str_calc_lat = "24N56." + str(calc_lat)
            str_calc_lon = "80W27." + str(calc_lon)
            setGPS(str_calc_lat, str_calc_lon, ind, tile) 
            print "Wrote metadata to image " + str(ind)
            print "Determined by interpolation"
            print "CALC LAT: " + str_calc_lat + " CALC LON: " + str_calc_lon
        elif good_data:
            tile = crop_frame(pil_im)
            setGPS(lat, lon, ind, tile)
            print "Wrote metadata to image " + str(ind)
            

        #if not good_data:
        #   final_img.save("/home/cavr/GPSImages/" + str(ind) + ".jpg")
        ind=ind+1
        
    #print str(ind)
    #print str(len(frame_list))
    #closes the stream and returns a list 
    cap.release()
    #cv2.waitKey(0)
    cv2.destroyAllWindows() 
    print "Frames processed...creating mosaic"
    
    #return frame_list, picture_list

if __name__ == "__main__":
    text_capture('/home/cavr/160721_Videos/191000.MPG')
    

    

