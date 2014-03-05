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
import matplotlib.pyplot as plt


directory = sys.argv[1]

images = [] #list of images
times = [] #list of timestamps
#dims = [] #list of image dimensions (w,h)
faces = [] #list of face regions (name,x,y,w,h)

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
    if filename == 'contacts.xml':
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

        # Get timestamp and photo dimensions
#        w = h = 0 # TODO dimensions maybe later
        for prop, value, opts in exif:
            if prop == 'exif:DateTimeOriginal':
                try:
                    times.append(time.mktime(time.strptime(value[:16], '%Y-%m-%dT%H:%M'))) #'2011-10-09T13:57:53'
                except ValueError:
                    # if original time missing, use modify time
                    t = xmp['http://ns.adobe.com/xap/1.0/'][0][1]
                    times.append(time.mktime(time.strptime(t[:16], '%Y-%m-%dT%H:%M')))
                break
#             if prop == 'exif:PixelXDimension':
#                 w = int(value)
#             if prop == 'exif:PixelYDimension':
#                 h = int(value)
#         dims.append((w, h))

        # Create "bag of faces" feature vector, weighted by what
        # fraction of the image the face occupies
        i = 1 #current region index
        faceVect = np.zeros(len(names))
        for prop, value, opts in regions:
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Name':
                f = names.index(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:w':
                w = float(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:h':
                h = float(value)
            if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:unit':
                i += 1
                faceVect[f] = w * h                
        faces.append(faceVect)

        # TODO what to do with this? Get face region information
#         i = 1 #current region index
#         name = None
#         x = y = 0.0 #these are fractions of the photo dimensions
#         w = h = 0.0
#         for prop, value, opts in regions:
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Name':
#                 name = value
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:x':
#                 x = float(value)
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:y':
#                 y = float(value)
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:w':
#                 w = float(value)
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:h':
#                 h = float(value)
#             if prop == 'mwg-rs:Regions/mwg-rs:RegionList[' + str(i) + ']/mwg-rs:Area/stArea:unit':
#                 faces.append((name, x, y, w, h))
#                 i += 1
#                 name = None
#                 x = y = 0.0
#                 w = h = 0.0

    except IOError:
        continue

# Save as .mat file
io.savemat('../data/April_full_fixedTime.mat', {'images': images,
                                              'timestamps': times,
#                                                  'dimensions': dims,
                                              'faces': faces}
           )
