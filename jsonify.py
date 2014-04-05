# April Shen
# Script to turn a csv file of metro map lines into a json file for d3

import sys, time
import numpy as np
import scipy.io as io
import scipy.misc as misc
import matplotlib.cbook as cbook

if __name__ == '__main__':
    filename = sys.argv[1]
    websitePath = '/Users/april/Dropbox/classes/CSE576/project/apriltuesday.github.io/'

    # Read in map
    map = []
    for l in open(filename + '.csv'):
        line = [int(i) for i in l.split(',')]
        map.append(line)
    nodes = list(set(cbook.flatten(map)))
    # Need these for times, faces, and images
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0][imgs]
    times = mat['timestamps'][imgs]
    faces = mat['facesBinary'][imgs]

    # Open json file
    f = open(websitePath + filename + '.json', 'w+')
    
    # Write node information
    f.write('{"nodes":[')
    strs = []
    index = 100
    for i, pic, date in zip(nodes, images, times):
        strs.append('{"id": "n' + str(i) + '", "date": "' + time.strftime('%a %bc %d %Y %H:%M:%S', time.localtime(date[0]))) + '", "x": ' + str(index) + ', "y": 300, "fixed": 0, "edges": [')

    f.close()
