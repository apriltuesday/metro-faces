# April Shen
# Metro maps X Photobios -- Script to turn a csv file of metro map lines
# into .graphml file that can be read by JS

import sys, time
import numpy as np
import scipy.io as io
import scipy.misc as misc
import matplotlib.cbook as cbook

filename = sys.argv[1]
websitePath = '/Users/april/Dropbox/classes/CSE576/project/apriltuesday.github.io/'

# Some boilerplate we need
header = '<?xml version="1.0" encoding="iso-8859-1"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"> <key id="x" for="node" attr.name="x coordinate" attr.type="int"/> <key id="y" for="node" attr.name="y coordinate" attr.type="int"/> <key id="label" for="node" attr.name="station name" attr.type="string"/> <key id="date" for="node" attr.name="date" attr.type="string"/> <key id="dummy" for="node" attr.name="dummy" attr.type="int"> <default>0</default> </key>'
middle = '<graph id="G" edgedefault="undirected">'
footer = '</graph></graphml>'

# Read in map
map = []
for l in open(filename + '.csv'):
    line = [int(i) for i in l.split(',')]
    map.append(line)

# Create graphml file of same name
f = open(websitePath + filename + '.graphml', 'w+')
f.write(header + '\n\n')

# Write line information
for i in range(len(map)):
    f.write('<key id="l' + str(i) + '" for="edge" attr.name="line ' + str(i))
    f.write('" attr.type="boolean" color.r="0" color.g="255" color.b="0" importance="10" title=""><default>FALSE</default></key>\n')

f.write('\n\n' + middle + '\n\n')

# Write node information
imgs = list(set(cbook.flatten(map)))
# Need these for times and displaying images
#mat = io.loadmat('../data/April_full_dataset_binary.mat')
mat = io.loadmat('../data/April_full_dataset_binary.mat')
images = mat['images'][:,0][imgs]
times = mat['timestamps'][imgs]

for i, pic, date in zip(imgs, images, times):
    f.write('<node id="n' + str(i) + '">')
    f.write('<data key="x">0</data> <data key="y">0</data> <data key="label">' + str(i) + '</data>')

    # Parse date into DD/MM/YYYY
    # TODO missing dates???
    d = time.strftime('%d/%m/%Y', time.localtime(date[0]))
    f.write('<data key="date">' + d + '</data> <data key="dummy">0</data></node>\n')

    # Also make the HTML image page
    imgPath = 'images/' + str(i) + '.png'
    misc.imsave(websitePath + imgPath, pic)
    page = open(websitePath + 'imagePages/' + str(i) + '.html', 'w+')
    page.write('<img src="' + imgPath + '">')
    page.close()

f.write('\n\n')

# Write edge information
# l0 is always connection info, in consecutive pairs
count = 0
for j in range(0, len(map[0])-1, 2):
    f.write('<edge id="e' + str(count) + '" ')
    f.write('source="n' + str(map[0][j]) + '" target="n' + str(map[0][j+1]) + '">')
    f.write('<data key="l0">true</data></edge>\n')
    count += 1
f.write('\n')    

# the other lines
for i in range(1, len(map)):
    line = map[i]
    for j in range(len(line)-1):
        f.write('<edge id="e' + str(count) + '" ')
        f.write('source="n' + str(line[j]) + '" target="n' + str(line[j+1]) + '">')
        f.write('<data key="l' + str(i) + '">true</data></edge>\n')
        count += 1
    f.write('\n')

f.write('\n' + footer)
f.close()
