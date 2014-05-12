# April Shen
# Social graph through time -- Main File
#!/usr/bin/python

from __future__ import division
import sys, time, json
import numpy as np
import scipy.io as io
import scipy.misc as misc
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from random import sample, choice, shuffle, random
from collections import Counter
import Queue
import pico

# Constraints of the map
NUM_PPL = 5

# Numbers of bins
NUM_CLUSTERS = 20
NUM_TIMES = 10
NUM_LOCS = 200

# For output files etc.
websitePath = '../apriltuesday.github.io/'


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
    maxVal = max(values)
    minVal = min(values)
    length = (maxVal - minVal) / k # length of interval

    # Compute binary vector for each item based on its value
    vectors = np.zeros((n, k))
    for i in items:
        if np.isfinite(values[i]):
            whichBin = int((values[i] - minVal) / length) - 1
            vectors[i, whichBin] = 1
    return vectors


def binValues(years, longitudes, latitudes):
    """
    Bin times and GPS coordinates.  If a value is missing, we assume
    it covers nothing (shouldn't happen if we correct invalid values
    beforehand).
    """
    times = bin(years, NUM_TIMES)
    times = times[:, ~np.all(times == 0, axis=0)]
    longs = bin(longitudes, NUM_LOCS)
    lats = bin(latitudes, NUM_LOCS)
    # just need place-bins, not individual long/lat
    places = np.zeros((years.shape[0], NUM_LOCS**2))
    nonLongs = np.nonzero(longs)
    nonLats = np.nonzero(lats)
    for img, lo, la in zip(nonLongs[0], nonLongs[1], nonLats[1]):
        places[img, lo + la*NUM_LOCS] = 1
    places = places[:, ~np.all(places == 0, axis=0)]
    return times, places


def clusterFaces(faces):
    """
    Cluster faces using group counts.
    """
    c = Counter()
    for img in np.arange(faces.shape[0]):
        group = tuple(sorted(np.nonzero(faces[img])[0]))
        if len(group) > 0 and len(group) < 20:
            c[group] += 1
    sortedClusters = sorted(c, key=lambda x: 1/(c[x] * len(x))) # by count and by size
    return sortedClusters


def fixInvalid(years, valid):
    """
    Fix invalid timestamps, by replacing with a random value generated
    from the distribution of valid times within years.
    """
    # TODO pick the appropriate distribution?
    mu = np.ma.mean(valid)
    sigma = np.ma.std(valid)
    a = np.ma.min(valid)
    b = np.ma.max(valid)
    correctMu = (mu - a) / (b - a)
    correctSigma = sigma / (b - a)**2
    alpha = correctMu * ((correctMu - correctMu**2) / correctSigma - 1)
    beta = (1 - correctMu) * alpha / correctMu
    years[valid.mask] = np.round(np.random.beta(alpha, beta, valid[valid.mask].shape) * (b-a) + a)
    return years


def correctTimestamps(years, faces, clusters):
    """
    Correct invalid timestamps, using face clusters to predict reasonable times.
    In our case, invalid times are Feb 2014 (hack)
    """
    timeObjs = [time.localtime(y) for y in years]
    invalid = [x.tm_year == 2014 and x.tm_mon == 2 for x in timeObjs]
    valid = np.ma.masked_where(invalid, years)
    clusters.reverse()
    for cl in clusters:
        pool = list(set(np.nonzero(faces[:,cl])[0])) #photos containing these faces
        if len(pool) > 0:
            years[pool] = fixInvalid(years[pool], valid[pool])
    clusters.reverse()


def saveGraph(filename, A, clusters, names, nodes):
    """
    Save graph defined by adjacency matrix A in a JSON file.
    Also store their cluster (if existant) and name.
    """
    f = open(filename, 'w+')
    clusterInd = {} #easier form to work with here
    for i in range(len(clusters)):
        for j in clusters[i]:
            # be a bit careful to deal with overlapping clusters
            if j in clusterInd.keys():
                clusterInd[j].append(i+1)
            else:
                clusterInd[j] = [i+1]
    strs = []
    f.write('{ "nodes": [\n')
    # Write nodes
    for node in nodes:
        strs.append('{"name": "' + names[node] + '", "group": ' + str(clusterInd[node] if node in clusterInd.keys() else [0]) + '}')
    f.write(',\n'.join(strs) + '],\n"links": [\n')
    strs = []
    # Write links
    xs, ys = np.nonzero(A)
    for i, j in zip(xs, ys):
        strs.append('{"source": ' + str(i) + ',  "target": ' + str(j) + ', "value": ' + str(A[i,j]) + '}')
    f.write(',\n'.join(strs) + '] }')
    f.close()


def coverage(paths, xs, weights):
    """
    Computes coverage of map, using xs as features weighted by weights.
    """
    subset = list(set(cbook.flatten(paths)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def greedy(subset, candidates, xs, weights):
    """
    Greedily choose max coverage candidate (faces covering images)
    and add to subset
    """
    maxCoverage = -float('inf')
    if len(candidates) == 0:
        return
    for p in candidates:
        c = coverage(subset + [p], xs, weights)
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
    subset.append(maxP)


def makeGraph(pool, faces, faceClusters):
    """
    Choose subset of faces to display in the graph, based on coverage
    of images.
    """
    subset = []
    xs = faces[pool].T
#    ppl = np.arange(xs.shape[0])
    ppl = faceClusters
    for i in np.arange(NUM_PPL):
        greedy(subset, ppl, xs, np.ones(xs.shape[1])) #uniform weights
    return subset

        
if __name__ == '__main__':
    # Load data
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0]
    faces = mat['facesBinary']
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()

    # Form hashtable of (group of faces) -> (photo count), and correct timestamps
    faceClusters = clusterFaces(faces)
    correctTimestamps(years, faces, faceClusters)
#    faceClusters = faceClusters[:NUM_CLUSTERS]

    # Bin times and locations
    times, places = binValues(years, longitudes, latitudes)
    NUM_TIMES = times.shape[1]

    # Create a social graph for each point in time
    for iter in np.arange(NUM_TIMES):
        prefix = 'newTest' + str(iter)
        pool = np.nonzero(times[:, iter])[0]
        clusters = makeGraph(pool, faces, faceClusters) #each is a cluster

        # Form adjacency matrix of social graph
        subset = list(set(cbook.flatten(clusters)))
        m = len(subset)
        A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in subset for j in subset]).reshape((m,m))
        # Save adjacency matrix to json
        saveGraph(websitePath + prefix + '-graph.json', A, clusters, names, subset)           
