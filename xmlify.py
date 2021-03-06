# April Shen
# Metro maps X Photobios -- Script to turn a csv file of metro map lines
# into .graphml file that can be read by JS

import sys, time
import numpy as np
import scipy.io as io
import scipy.misc as misc
import matplotlib.cbook as cbook

filename = sys.argv[1]
websitePath = '/Users/april/Dropbox/project/apriltuesday.github.io/'

# Some boilerplate we need
header = '<?xml version="1.0" encoding="iso-8859-1"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns"xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"> <key id="x" for="node" attr.name="x coordinate" attr.type="int"/> <key id="y" for="node" attr.name="y coordinate" attr.type="int"/> <key id="label" for="node" attr.name="station name" attr.type="string"/> <key id="date" for="node" attr.name="date" attr.type="string"/> <key id="dummy" for="node" attr.name="dummy" attr.type="int"> <default>0</default> </key>'
middle = '<graph id="G" edgedefault="undirected">'
footer = '</graph></graphml>'

# Read in map
map = []
for l in open(filename + '.csv'):
    line = [int(i) for i in l.split(',')]
    map.append(line)

# First line in the file is actually links
#links = map[0]
#numLinks = len(links) / 2
#map = map[1:]

# Create graphml file of same name
f = open(websitePath + filename + '.graphml', 'w+')
f.write(header + '\n\n')

# Write link/connection information
#for i in range(numLinks):
#    f.write('<key id="link' + str(i) + '" for="edge" attr.name="link ' + str(i))
#    f.write('" attr.type="boolean" color.r="0" color.g="255" color.b="0" importance="10" title=""><default>FALSE</default></key>\n')
#f.write('\n')

# Write line information
for i in range(len(map)):
    f.write('<key id="l' + str(i) + '" for="edge" attr.name="line ' + str(i))
    f.write('" attr.type="boolean" color.r="0" color.g="255" color.b="0" importance="10" title=""><default>FALSE</default></key>\n')

f.write('\n\n' + middle + '\n\n')

# Write node information
imgs = list(set(cbook.flatten(map)))
# Need these for times, faces, and images
mat = io.loadmat('../data/April_full_new.mat')
images = mat['images'][:,0][imgs]
times = mat['timestamps'][imgs]
faces = mat['facesBinary'][imgs]

# Get a master list of names from contacts.xml
names = []
for line in open('../data/faces/contacts.xml'):
    try:
        rest = line[line.index('name=')+6:]
        name = rest[:rest.index('"')]
        names.append(name)
    except ValueError:
        continue

for i, pic, date, ppl in zip(imgs, images, times, faces):
    f.write('<node id="n' + str(i) + '">')
    f.write('<data key="x">0</data> <data key="y">0</data> <data key="label">' + str(i) + '</data>')

    # Parse date into DD/MM/YYYY
    d = time.strftime('%d/%m/%Y', time.localtime(date[0]))
    f.write('<data key="date">' + d + '</data> <data key="dummy">0</data></node>\n')

    # Also make the HTML image page
    imgPath = 'images/' + str(i) + '.png'
    misc.imsave(websitePath + imgPath, pic)
    page = open(websitePath + 'imagePages/' + str(i) + '.html', 'w+')
    page.write('<img src="' + imgPath + '">\n')
    d = time.strftime('%Y-%m-%d', time.localtime(date[0]))
    page.write('<strong>' + d + '</strong><br>')
    for j in range(len(ppl)):
        if ppl[j] == 1:
            page.write(names[j])
            if (ppl[j+1:] != 0).any(): #not the end
                page.write(', ')
    page.close()

f.write('\n\n')

# Write edge information
# first line is always connection info, in consecutive pairs
count = 0
#which = 0 #which link
#for j in range(0, len(links)-1, 2):
#    f.write('<edge id="e' + str(count) + '" ')
#    f.write('source="n' + str(links[j]) + '" target="n' + str(links[j+1]) + '">')
#    f.write('<data key="link' + str(which) + '">true</data></edge>\n')
#    which += 1
#    count += 1
#f.write('\n')    

# the other lines
for i in range(len(map)):
    line = map[i]
    for j in range(len(line)-1):
        f.write('<edge id="e' + str(count) + '" ')
        f.write('source="n' + str(line[j]) + '" target="n' + str(line[j+1]) + '">')
        f.write('<data key="l' + str(i) + '">true</data></edge>\n')
        count += 1
    f.write('\n')

f.write('\n' + footer)
f.close()
