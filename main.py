# April Shen
# Metro maps X Photobios - Main file
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

# Constraints of the map
NUM_LINES = 10
NUM_PHOTOS = 20 #always equal to num-times?
TAU = 0.7 # This is the minimum coherence constraint

# Numbers of bins
NUM_CLUSTERS = 200
NUM_TIMES = NUM_PHOTOS
NUM_LOCS = 200

# For output files etc.
websitePath = '../apriltuesday.github.io/'
prefix = 'test'


def coverage(map, xs, weights):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = list(set(cbook.flatten(map)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def coherence(chain, faces):
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


def greedy(map, path, candidates, xs, weights, times):
    """
    Greedily choose the max-coverage candidate and add to map.
    """
    maxCoverage = -float('inf') #note that if weights are negative, we can have negative coverage!
    maxP = float('inf')
    total = list(set(cbook.flatten(map))) + path
    for p in candidates:
        if (times[path].dot(times[p]) > 0).any(): #same time bin as another photo in the path
            continue
        c = coverage(total + [p], xs, weights)
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
    if np.isfinite(maxP):
        candidates.remove(maxP) #XXX
        path.append(maxP)


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


def orderLines(paths, faceClusters):
    """
    Order lines and clusters according to shared images (in place).
    """
    # This makes the visualization easier and is kind of a huge hack
    # XXX maybe want to sort by shared faces?  time?
    i = 0
    while i < NUM_LINES:
        maxInt = 0
        maxJ = 0
        for j in range(len(paths[i+1:])):
            intersect = len(set(paths[i]) & set(paths[j]))
            if intersect > maxInt: #largest intersection
                maxJ = j
                maxInt = intersect
        if maxInt > 0: #swap
            temp = paths[maxJ]
            paths[maxJ] = paths[i+1]
            paths[i+1] = temp
            # also swap corresponding face clusters
            temp = faceClusters[maxJ]
            faceClusters[maxJ] = faceClusters[i+1]
            faceClusters[i+1] = temp
        i += 1


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


def binValues(years, longitudes, latitudes):
    """
    Bin times and GPS coordinates.  If a value is missing, we assume
    it covers nothing (shouldn't happen if we correct invalid values
    beforehand).
    """
    times = bin(years, NUM_TIMES)
    times = times[:, ~np.all(times == 0, axis=0)]
    #XXX invalid photos should cover no times
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


def fixInvalid(years, valid):
    """
    Fix invalid timestamps, by replacing with a random value generated
    from the distribution of valid times within years.
    """
    # XXX pick the appropriate distribution?
    mu = np.ma.mean(valid)
    sigma = np.ma.std(valid)
    a = np.ma.min(valid)
    b = np.ma.max(valid)
    correctMu = (mu - a) / (b - a)
    correctSigma = sigma / (b - a)**2
    alpha = correctMu * ((correctMu - correctMu**2) / correctSigma - 1)
    beta = (1 - correctMu) * alpha / correctMu
    #np.round(np.random.normal(mu, sigma, valid[valid.mask].shape))
    #np.round(np.random.uniform(np.ma.min(valid), np.ma.max(valid), valid[valid.mask].shape))
    years[valid.mask] = np.round(np.random.beta(alpha, beta, valid[valid.mask].shape) * (b-a) + a)
    return years


def correctTimestamps(years, clusters):
    # Correct invalid timestamps
    # In our case these are Feb 2014 (hack)
    timeObjs = [time.localtime(y) for y in years]
    invalid = [x.tm_year == 2014 and x.tm_mon == 2 for x in timeObjs]
    valid = np.ma.masked_where(invalid, years)
    clusters.reverse()
    for cl in clusters:
        pool = list(set(np.nonzero(faces[:,cl])[0])) #photos containing these faces
        if len(pool) > 0:
            years[pool] = fixInvalid(years[pool], valid[pool])
    clusters.reverse()
    

def getWeights(faces, times, places):
    """
    Weight importance of faces, times, places by frequency (normalized).
    """
    #faceWeights = np.apply_along_axis(np.count_nonzero, 0, faces)
    faceWeights = np.apply_along_axis(np.sum, 0, faces)
    faceWeights = faceWeights / np.linalg.norm(faceWeights)
    timeWeights = np.apply_along_axis(np.sum, 0, times)
    timeWeights = timeWeights / np.linalg.norm(timeWeights)
    placeWeights = np.apply_along_axis(np.sum, 0, places)
    placeWeights = placeWeights / np.linalg.norm(placeWeights)
    return np.hstack([faceWeights, timeWeights, placeWeights])


def saveMap(filename, paths, connections, images):
    """
    Save map in a JSON file. Also save the corresponding photos.
    """
    f = open(filename, 'w+')
    nodes = list(set(cbook.flatten(paths)))
    pathInd = {} #easier form to work with here
    for i in range(len(paths)):
        for j in paths[i]:
            if j not in pathInd.keys(): # XXX overlapping clusters?
                pathInd[j] = i+1
    strs = []

    # Write nodes
    f.write('{ "nodes": [\n')
    for node in nodes:
        imgPath = 'images/' + str(node) + '.png'
        misc.imsave(websitePath + imgPath, images[node])
        strs.append('{"id": ' + str(node) + ', "line": ' + str(pathInd[node]) + '}')
    f.write(',\n'.join(strs) + '],\n"links": [\n')
    strs = []

    # Write links
    for i in range(len(paths)):
        p = paths[i]
        for j in range(0, len(p)-1):
            strs.append('{"source": ' + str(nodes.index(p[j])) + ',  "target": ' + str(nodes.index(p[j+1])) + ', "line": ' + str(i+1) + '}')
    f.write(',\n'.join(strs) + ']}')
    f.close()


def saveGraph(filename, A, clusters, names):
    """
    Save graph defined by adjacency matrix A in a JSON file.
    Also store their cluster (if existant) and name.
    """
    f = open(filename, 'w+')
    nodes = np.arange(A.shape[0])
    clusterInd = {} #easier form to work with here
    for i in range(len(clusters)):
        for j in clusters[i]:
            clusterInd[j+1] = i+1 # XXX overlapping clusters?
    strs = []
    f.write('{ "nodes": [\n')
    # Write nodes
    for node in nodes:
        strs.append('{"name": "' + names[node] + '", "group": ' + str(clusterInd[node] if node in clusterInd.keys() else 0) + '}')
    f.write(',\n'.join(strs) + '],\n"links": [\n')
    strs = []
    # Write links
    xs, ys = np.nonzero(A)
    for i, j in zip(xs, ys):
        strs.append('{"source": ' + str(i) + ',  "target": ' + str(j) + ', "value": ' + str(A[i,j]) + '}')
    f.write(',\n'.join(strs) + '] }')
    f.close()


def saveFeatures(filename, faces, dates, longs, lats):
    """
    Save feature vectors in a JSON file.
    """
    f = open(filename, 'w+')
    data = {'faces': faces.tolist(), 'dates': dates.tolist(), 'longs': longs.tolist(), 'lats': lats.tolist()}
    json.dump(data, f)
    f.close()


def makeMap(prefix, faces, years, longitudes, latitudes, photos):
    """
    Do everything. (enables zooming)
    Photos is the list of indices of images that we're currently choosing from.
    Prefix is a file prefix to use for the map
    faces, years, etc. are the full feature matrices (XXX eventually not)
    """
    if len(photos) == 0:
        return
    # Bin times and GPS coordinates
    times, places = binValues(years, longitudes, latitudes)

    # Choose face clusters
    #TODO we don't want to redo this work every time
    faceClusters = clusterFaces(faces)[:NUM_LINES]

    ########### CREATING THE MAP ###############

    vects = np.hstack([faces, times, places])
    paths = []
    # For each face cluster, get high-coverage coherent path for its photos
    for cl in faceClusters:
        #XXX figure out weighting & coherence-like constraints

        # choose weights for this cluster's line, de-emphasizing faces outside the cluster
        other = np.empty(faces.shape)
        other[:, cl] = faces[:, cl]
        ind = list(set(ppl) - set(cl))
        other[:, ind] = -faces[:,ind]
        weights = getWeights(other, times, places)

        # choose a pool of photos containing at least len(cl)*TAU faces within cl
        nonz = np.nonzero(faces[:,cl])[0]
        sumz = dict(zip(nonz, np.apply_along_axis(np.count_nonzero, 1, faces[nonz][:,cl])))
        pool = filter(lambda x: x in sumz.keys() and sumz[x]>len(cl)*TAU, photos)

        # choose a path greedily from among the pool
        path = []
        for i in range(NUM_PHOTOS):
            greedy(paths, path, pool, vects, weights, times)
        paths.append(path)

    # Fix lines to show overlaps in faces
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i == j:
                continue
            for img in paths[j]:
                if sum(faces[img, faceClusters[i]]) == len(faceClusters[i]): #img includes all the faces in i
                    paths[i].append(img)

    # Sort and re-order lines to improve layout
    paths = [sorted(x, key=lambda x: years[x]) for x in paths]
    orderLines(paths, faceClusters)

    # Save map to json
    saveMap(websitePath + prefix + '-map.json', paths, [], images)


if __name__ == '__main__':

    ########### LOADING THE DATA ###############
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0]
    faces = mat['facesBinary'] #mat['faces]
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()

    ########### PROCESSING THE DATA ###############

    # Omit people who don't appear
    names = names[~np.all(faces == 0, axis=0)]
    faces = faces[:, ~np.all(faces == 0, axis=0)]
    n, m = faces.shape
    photos = np.arange(n)
    ppl = np.arange(m)

    # Form adjacency matrix of the social graph (including ME)
    A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in ppl for j in ppl]).reshape((m,m))
    # now omit ME
    faces = faces[:,1:]
    ppl = np.arange(m-1)

    # Form hashtable of (group of faces) -> (photo count)
    clusters = clusterFaces(faces) #XXX
    # Use these to correct invalid times
    correctTimestamps(years, clusters)

    # Recursively make maps at all levels of zoom
    # NOTE: THIS IS STUPID, WE DO NOT WANT THIS
    makeMap(prefix, faces, years, longitudes, latitudes, photos)

    ########### SAVING THE DATA ###############
    faceClusters = []
    # Save adjacency matrix to json
    saveGraph(websitePath + prefix + '-graph.json', A, faceClusters, names)
    
    # Save feature vectors to json
    np.place(years, np.isinf(years), np.min(years))
    np.place(longitudes, np.isinf(longitudes), 0)
    np.place(latitudes, np.isinf(latitudes), 0)
    saveFeatures(websitePath + prefix + '-feats.json', faces, years, longitudes, latitudes)
