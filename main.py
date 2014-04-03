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

# Constraints of the map
NUM_LINES = 5
NUM_PHOTOS = 5
TAU = 0.2 # This is the minimum coherence constraint

# Numbers of bins
NUM_CLUSTERS = 150
NUM_TIMES = 100
NUM_LOCS = 50


def coverage(map, xs, weights):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = list(set(cbook.flatten(map)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def coherence(chain, faces, times):
    """
    Compute the coherence of the given chain.
    Coherence is based on what faces are included and chronology.
    """
    # Coherence is min. number of faces shared between two
    # images, weighted by frequency. also penalized if time moves backwards.
    minShare = float('inf')
    for i in np.arange(len(chain)-1):
        pic1 = chain[i]
        pic2 = chain[i+1]
        #faces shared by both
        numShare = faces[pic1].dot(faces[pic2])
        #if time moves backwards, penalize XXX hack alert
#        if (times[pic1] != 0).any() and (times[pic2] != 0).any:
#            if np.nonzero(times[pic1])[0][0] > np.nonzero(times[pic2])[0][0]:
#                numShare /= 2.0
        if numShare < minShare:
            minShare = numShare
    return minShare


def connectivity(map, faces):
    """
    Compute connectivity of the map. Two lines are considered
    connected if their nodes share faces (and maybe places/times?)
    """
    if len(map) < 2:
        return 0
    total = 0
    numLines = len(map)
    # Flatten each line of the map
    flatMap = []
    for u in np.arange(numLines):
        flatMap.append(list(set(cbook.flatten(map[u]))))

    # Count number of lines that intersect
    for u in np.arange(numLines):
        for v in np.arange(u+1, numLines):
            if np.logical_and(faces[flatMap[u]].sum(axis=0) != 0, faces[flatMap[v]].sum(axis=0) != 0).any(): #intersect
                total += 1
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
#     args = sys.argv
#     if len(args) < 2:
#         k = 10
#     else:
#         k = int(args[1]) # number of year bins to select; ultimately a function of zooming/the UI

    # Load data
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0]
    faces = mat['faces'] + mat['facesBinary'] 
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()

    # Omit people who don't appear
    names = names[~np.all(faces == 0, axis=0)]
    faces = faces[:, ~np.all(faces == 0, axis=0)]
    n, m = faces.shape
    photos = np.arange(n)
    ppl = np.arange(m)

    # Bin times and GPS coordinates. If value is missing we assume it covers nothing.
    times = bin(years, NUM_TIMES)
    times = times[:, ~np.all(times == 0, axis=0)]
    longs = bin(longitudes, NUM_LOCS)
    lats = bin(latitudes, NUM_LOCS)
    # just need place-bins
    places = np.zeros((n, NUM_LOCS**2))
    nonLongs = np.nonzero(longs)
    nonLats = np.nonzero(lats)
    for img, lo, la in zip(nonLongs[0], nonLongs[1], nonLats[1]):
        places[img, lo + la*NUM_LOCS] = 1
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
    plt.figure(0)
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

    # omit ME
    # note we include me for the social graph but not for like everything else
    faces = faces[:,1:]
    faceWeights = faceWeights[1:]

    # Cluster faces
    c = cluster.bicluster.SpectralCoclustering(n_clusters=NUM_CLUSTERS, svd_method='arpack')
    c.fit(A[1:, 1:]) #omit ME
    clusters = [] #list of lists, each of which lists face indices in that cluster
    for i in range(NUM_CLUSTERS):
        clusters.append(c.get_indices(i)[0])

    # Choose clusters of faces to use for lines
    whichClusters = []
    for i in range(NUM_LINES):
        greedy(whichClusters, clusters, faces.T, np.ones(n))
    for i in range(NUM_LINES):
        for j in whichClusters[i]:
            print j+1, names[j+1]
        print '------'

    vects = np.hstack([faces, times, places])
    weights = np.hstack([faceWeights, timeWeights, placeWeights])
    map = []
    iter = 1
    # For each face cluster, get high-coverage coherent path for its photos
    for cl in whichClusters:
        pool = list(set(np.nonzero(faces[:,cl])[0])) #photos containing these faces
        path = []

        for i in range(NUM_PHOTOS):
            if len(pool) == 0:
                break
            greedy(path, pool, vects, weights)
            # throw out photos not coherent with path
            newPool = []
            for img in pool:
                if img not in path and coherence(path + [img], faces, times) > TAU:
                    newPool.append(img)
            pool = newPool
        map.append(sorted(path, key=lambda x: np.nonzero(times[x])[0][0]))

        print 'done with iteration', iter
        iter += 1

    # Display paths
    for i in range(NUM_LINES):
        plt.figure(i+1)
        path = map[i]
        for j, img in zip(range(len(path)), path):
            plt.subplot(1, len(path), j+1)
            plt.title('image ' + str(img))
            plt.imshow(images[img])

    plt.show()
