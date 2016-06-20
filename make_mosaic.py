import cv2
import numpy as np
from PIL import Image, ImageDraw
from frame import Frame

def insertion_sort(items):
        """ Implementation of insertion sort """
        for i in range(1, len(items)):
                j = i
                while j &gt; 0 and items[j] &lt; items[j-1]:
                        items[j], items[j-1] = items[j-1], items[j]
                        j -= 1

def mosaic(frame_list):

    
    lat_dec_list = []
    lon_dec_list = []

    for frame in frame_list:
	lat_dec = frame.lat[6:]
	lon_dec = frame.lon[6:]
	lat_dec_list.append(lat_dec)
	lon_dec_list.append(lon_dec)

    maximum_lat = max(lat_dec_list)
    mininum_lat = min(lat_dec_list)
    maximum_lon = max(lon_dec_list)
    minimum_lon = min(lon_dec_list)

    lat_range = maximum_lat-minimum_lat
    lon_range = maximum_lon-minimum_lon

    size = (lon_range*720,lat_range*480)

    blank = Image.new('RGB',size,(0,0,0))


    shape = (log_range,lat_range)

    frame_array = np.empty(shape,Frame,'C')

    for frame in frame_list:
	lat_dec = frame.lat[6:]
	lon_dec = frame.lon[6:]
	frame_array[(lat_dec-minimum_lat),(maximum_lon-(lat_dec-minimum_lon))] = frame

>>> it = np.nditer(frame_array, flags=['c_index'])
>>> while not it.finished:
...     print "%d <%d>" % (it[0], it.index),
...     it.iternext()

    i = 0
    j = 0
    while(i < len(frame_array)):
	while(j < len(frame_array[0]):
	    place = ((i*720),(j*480))
	    img = Image.fromarray(frame_array[i,j].img)
	    blank.paste(img,place)

    blank.save('mosaic.png')


