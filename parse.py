# April Shen
# Metro maps X Photobios - Parse images into .mat format
#!/usr/bin/python

from __future__ import division
import os, sys, time
import numpy as np
import scipy.io as io
import scipy.ndimage as img
import scipy.misc as misc
from libxmp import utils


directory = sys.argv[1]

images = [] #list of images
times = [] #list of timestamps
longs = [] # list of longitude
lats = [] #list of latitude
faces = [] #list of face vectors (weights by region size)
facesBinary = [] #ditto but binary
faceRegions = [] #list of region information

# Get a master list of names from contacts.xml
names = []
for line in open(directory + '/contacts.xml'):
    try:
        rest = line[line.index('name=')+6:]
        name = rest[:rest.index('"')]
        names.append(name)
    except ValueError:
        continue

invalid = 0
# Parse images
for filename in os.listdir(directory):
    if filename == 'contacts.xml' or filename == '.DS_Store' or filename == '.picasa.ini':
        continue
    try:
        # Get image, downsample if too large
        f = directory + '/' + filename
        image = img.imread(f)
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            image = misc.imresize(image, 0.5)
        images.append(image)

        # Get metadata
        xmp = utils.file_to_dict(f)
        regions = xmp['http://www.metadataworkinggroup.com/schemas/regions/']
        exif = xmp['http://ns.adobe.com/exif/1.0/']

        # Get timestamp and location
        gpsFound = False
        for prop, value, opts in exif:
            if prop == 'exif:PixelXDimension':
                width = int(value)
            if prop == 'exif:PixelYDimension':
                height = int(value)

            if prop == 'exif:DateTimeOriginal':
                try:
                    times.append(time.mktime(time.strptime(value[:16], '%Y-%m-%dT%H:%M'))) #'2011-10-09T13:57:53'
                except ValueError:
                    try:
                        # if original time missing, use modify time
                        t = xmp['http://ns.adobe.com/xap/1.0/'][0][1]
                        times.append(time.mktime(time.strptime(t[:16], '%Y-%m-%dT%H:%M')))
                    except ValueError:
                        # still missing, just set to inf
                        times.append(float('inf'))

            if prop == 'exif:GPSLongitude':
                gpsFound = True
                # Round to nearest degree, set south and west to negative
                degree = float(value[:value.index('.')].replace(',','.'))
                if value[-1] == 'W':
                    degree = -degree
                longs.append(degree)
            if prop == 'exif:GPSLatitude':
                gpsFound = True
                # Round to nearest degree, set south and west to negative
                degree = float(value[:value.index('.')].replace(',','.'))
                if value[-1] == 'S':
                    degree = -degree
                lats.append(degree)

        # If no GPS, set long/lat to inf
        if not gpsFound:
            longs.append(float('inf'))
            lats.append(float('inf'))
                

        # Create "bag of faces" feature vector
        # Two versions: one weighted by region size, the other binary
        i = 1 #current region index
        w = h = 0 #dimension of region
        x = y = 0 #location of region
        faceVect = np.zeros(len(names))
        faceVectBin = np.zeros(len(names))
        regionVect = [(0,0,0,0) for nam in range(len(names))] #(x,y,w,h) for each name
        for prop, value, opts in regions:
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Name':
                f = names.index(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:w':
                w = float(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:h':
                h = float(value)
            # region location
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:x':
                x = float(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:y':
                y = float(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:unit':
                i += 1
                faceVect[f] = w * h
                faceVectBin[f] = 1
                regionVect[f] = (int(x*width), int(y*height), int(w*width), int(h*height)) # convert to absolute pixel
        faces.append(faceVect)
        facesBinary.append(faceVectBin)
        faceRegions.append(regionVect)

    except IOError:
        continue

# Save as .mat file
io.savemat('../data/April_full_gps.mat', {'images': images,
                                          'timestamps': times,
                                          'longitudes': longs,
                                          'latitudes': lats,
                                          'faces': faces,
                                          'facesBinary': facesBinary}
           )

# store region info in a CSV file so C++ can use it
output = open('../data/regions.csv', 'w+')
for i in range(len(faceRegions)):
    output.write('id ' + str(i) + '\n')
    for tup in faceRegions[i]:
        output.write(','.join([str(x) for x in tup]) + '\n')
output.close()
