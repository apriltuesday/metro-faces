# April Shen
# Metro maps X Photobios - Main file
#!/usr/bin/python

from __future__ import division
import sys
import numpy as np
import scipy.io as io
#import scipy.cluster as cluster
import networkx as nx
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from random import sample, choice, shuffle, random
import Queue


NUM_LINES = 4
NUM_CLUSTERS = 80 #???

def coverage(map, xs, weights):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = list(set(cbook.flatten(map)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def greedy(map, candidates, xs, weights):
    """
    Greedily choose the max-coverage candidate and add to map.
    """
    maxCoverage = 0.0
    maxP = 0
    for p in candidates:
        c = coverage(map + [p], xs, weights)
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
    map.append(maxP)


def getNames():
    """
    Get a master list of names from contacts.xml
    """
    names = []
    for line in open('../data/faces/contacts.xml'):
        try:
            rest = line[line.index('name=')+6:]
            name = rest[:rest.index('"')]
            names.append(name)
        except ValueError:
            continue
    return np.array(names)


def bin(values, k):
    """
    Bin values into k bins and returns binary vector indicating bin
    membership. Assumes we can sort.  May be infs but no nans.
    """
    n = values.shape[0]
    items = np.arange(n)

    # sorted items that have a valid value
    sortedItems = sorted(items[np.isfinite(values)], key=lambda i: values[i])
    # unique values, sorted
    sortedVals = sorted(set(values))
    # number of values per bin
    num = int(np.ceil(len(sortedVals) / k))
    bins = [sortedVals[x:x+num] for x in range(0, len(sortedVals), num)]

    # Compute binary vector for each item based on its value
    vectors = np.zeros((n, len(bins)))
    for i in items:
        if np.isfinite(values[i]):
            whichBin = map(lambda x: values[i] in x, bins).index(True)
            vectors[i, whichBin] = 1
    return vectors


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        k = 10
    else:
        k = int(args[1]) # number of year bins to select; ultimately a function of zooming/the UI

    # Load data
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0]
    faces = mat['faces'] + mat['facesBinary'] 
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()
    print 'done loading'

    # Omit people who don't appear
    names = names[~np.all(faces == 0, axis=0)]
    faces = faces[:, ~np.all(faces == 0, axis=0)]
    n, m = faces.shape
    photos = np.arange(n)
    ppl = np.arange(m)

    # Bin times and GPS coordinates. If value is missing we assume it covers nothing.
    times = bin(years, k)
    times = times[:, ~np.all(times == 0, axis=0)]
    longs = bin(longitudes, k)
    lats = bin(latitudes, k)
    # just need place-bins
    places = np.zeros((n, k**2))
    nonLongs = np.nonzero(longs)
    nonLats = np.nonzero(lats)
    for img, lo, la in zip(nonLongs[0], nonLongs[1], nonLats[1]):
        places[img, lo + la*k] = 1
    places = places[:, ~np.all(places == 0, axis=0)]

    # Weight importance of faces, times, places by frequency... normalize
    faceWeights = np.apply_along_axis(np.count_nonzero, 0, faces)
    faceWeights = faceWeights / np.linalg.norm(faceWeights)
    timeWeights = np.apply_along_axis(np.count_nonzero, 0, times)
    timeWeights = timeWeights / np.linalg.norm(timeWeights)
    placeWeights = np.apply_along_axis(np.count_nonzero, 0, places)
    placeWeights = placeWeights / np.linalg.norm(placeWeights)

    # Form adjacency matrix of the social graph
    A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in ppl for j in ppl]).reshape((m,m))

    # Graph that sucker
    G = nx.Graph(A)
    nodes = [A[i,i] for i in ppl]
    edges = []
    for (u,v) in G.edges():
        edges.append(A[u,v])

    #colors = cm.get_cmap(name='Spectral')
    #nx.draw(G, node_color=nodes, edge_color=edges, cmap=colors, edge_cmap=colors)
    layout = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=layout, node_size=[x*2 for x in nodes])
    nx.draw_networkx_edges(G, pos=layout, alpha=0.5, node_size=0, width=edges, edge_color='b')

    # Cluster faces
    c = cluster.bicluster.SpectralCoclustering(n_clusters=NUM_CLUSTERS)
    c.fit(A)
    clusters = [] #list of lists, each of which lists face indices in that cluster
    for i in range(NUM_CLUSTERS):
        clusters.append(c.get_indices(i)[0])

    # Choose clusters of faces to use for lines
    whichClusters = []
    for i in range(NUM_LINES):
        greedy(whichClusters, clusters, faces.T, np.ones(n))
    for i in range(NUM_LINES):
        for j in whichClusters[i]:
            print j, names[j]
        print '------'

    # For each face cluster, choose representative photos
    map = []
    vects = np.hstack([faces, times, places])
    weights = np.hstack([faceWeights, timeWeights, placeWeights])
    for cl in whichClusters:
        # get photos that contain faces in this cluster
        pool = set(np.nonzero(faces[:,cl])[0])
        greedy(map, pool, vects, weights)
        # TODO want to cover as many faces in this cluster as possible

    # Display
    plt.figure(0)
    for i in range(NUM_LINES):
        img = map[i]
        plt.subplot(1, NUM_LINES, i+1)
        plt.title('image ' + str(img))
        plt.imshow(images[img])

    plt.show()
