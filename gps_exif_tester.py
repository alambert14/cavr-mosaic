#!/usr/bin/env python
from PIL import Image
import piexif

im = Image.open("0001.jpg")
exif_dict = piexif.load(im.info["exif"])

exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = ((24, 1), (56972, 1000), (0, 1))
exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N'

exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = ((80, 1), (27127, 1000), (0, 1))
exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'W'

exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = ( () , () , () )
exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0

exif_bytes = piexif.dump(exif_dict)
im.save('sampleout.jpg', "jpeg", exif=exif_bytes)
