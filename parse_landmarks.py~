# April Shen
# Metro maps X Photobios - Parse landmarks into .mat format
#!/usr/bin/python

from __future__ import division
import os, sys, time
import numpy as np
import scipy.io as io
import scipy.misc as misc
from libxmp import utils


directory = '../data/faces/'
fn = sys.argv[1]
landmarks = [] # dimension (# photos) x (# faces) x (# landmarks) x 2 (coordinates)
NUM_LANDMARKS = 49

# First get image dimensions
dims = []
for filename in os.listdir(directory):
    if filename == 'contacts.xml' or filename == '.DS_Store' or filename == '.picasa.ini':
        continue
    try:
        # Get metadata
        f = directory + '/' + filename
        xmp = utils.file_to_dict(f)
        regions = xmp['http://www.metadataworkinggroup.com/schemas/regions/']
        exif = xmp['http://ns.adobe.com/exif/1.0/']
        for prop, value, opts in exif:
            if prop == 'exif:PixelXDimension':
                width = int(value)
            if prop == 'exif:PixelYDimension':
                height = int(value)
        dims.append((width, height))
    except IOError:
        continue

# Then parse landmarks file
i = 1
img = 0
faceVect = []
photoVect = []
for line in open(fn):
    if '.jpg' in line:
        w, h = dims[img]
        img += 1
        landmarks.append(photoVect)
        photoVect = []
        continue

    coords = [int(x) for x in line.split(',')]
    faceVect.append([coords[0] / w, coords[1] / h] if coords[0] > 0 else [-1,-1])
    if i == NUM_LANDMARKS:
        i = 1
        photoVect.append(faceVect)
        faceVect = []
    else:
        i += 1

# Save as .mat file
io.savemat('../data/landmarks.mat', {'landmarks': landmarks[1:] } )
