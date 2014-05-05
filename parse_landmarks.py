# April Shen
# Metro maps X Photobios - Parse landmarks into .mat format
#!/usr/bin/python

from __future__ import division
import os, sys, time
import numpy as np
import scipy.io as io
import scipy.misc as misc


filename = sys.argv[1]
landmarks = [] # dimension (# photos) x (# faces) x (# landmarks) x 2 (coordinates)
NUM_LANDMARKS = 49

i = 1
faceVect = []
photoVect = []
for line in open(filename):
    if '.jpg' in line:
        landmarks.append(photoVect)
        photoVect = []
        continue
    
    faceVect.append([int(x) for x in line.split(',')])
    if i == NUM_LANDMARKS:
        i = 1
        photoVect.append(faceVect)
        faceVect = []
    else:
        i += 1

# Save as .mat file
io.savemat('../data/landmarks2.mat', {'landmarks': landmarks } )
