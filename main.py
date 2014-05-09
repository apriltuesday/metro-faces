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
import pico

# Constraints of the map
NUM_LINES = 10
TAU = 0.8 # This is the minimum coherence constraint

# Numbers of bins
NUM_CLUSTERS = 200
NUM_TIMES = 50
NUM_LOCS = 200

# For output files etc.
websitePath = '../apriltuesday.github.io/'
prefix = 'test'


################### CORE ALGORITHM ########################


def coverage(map, xs, weights):
    """
    Computes coverage of map, using xs as features weighted by weights.
    """
    subset = list(set(cbook.flatten(map)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def coherence(path, photos, xs, times):
    """
    Return pool of photos sufficiently coherent with current path,
    based on features xs.
    times forces unique photo per time bin per path (and eventually, ordering)
    """
    # XXX WAIT THIS IS STUPID this should be a function eval not this crap
    pool = []
    for p1 in photos:
        add = True
        if (times[path].dot(times[p1]) > 0).any(): #same time bin as another photo in the path
            continue
        for p2 in path:
            # TODO: what exactly do we want here? norm? dot prod?
            dist = np.linalg.norm(xs[p1] - xs[p2])
            if dist > 2*TAU:
                add = False
                break
        if add:
            pool.append(p1)
    return pool


def greedy(map, path, candidates, xs, weights):
    """
    Greedily choose the candidate with max coverage relative to map+path,
    and add to path. Uses xs as features with the given weights.
    """
    maxCoverage = -float('inf') #note that if weights are negative, we can have negative coverage!
    if len(candidates) == 0:
        return
    totalMap = list(set(cbook.flatten(map))) + path
    for p in candidates:
        c = coverage(totalMap + [p], xs, weights)
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
    path.append(maxP)


################### DATA PROCESSING ########################


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
    # TODO maybe want to sort by shared faces?  time?
    i = 0
    while i < len(paths):
        maxInt = 0
        maxJ = 0
        for j in range(i+1, len(paths)):
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
    if num == 0:
        bins = [[x] for x in sortedVals]
    else:
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


################### DATA LOAD / SAVE ########################


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


def saveMap(filename, paths, images, faces, years, longs, lats):
    """
    Save map in a JSON file. Also save the corresponding photos.
    Store feature data as attributes of each node.
    """
    f = open(filename, 'w+')
    nodes = list(set(cbook.flatten(paths)))
    pathInd = {} #easier form to work with here
    for i in range(len(paths)):
        for j in paths[i]:
            if j in pathInd.keys():
                pathInd[j].append(i+1)
            else:
                pathInd[j] = [i+1]
    strs = []

    # Write nodes
    f.write('{ "nodes": [\n')
    for node in nodes:
        imgPath = 'images/' + str(node) + '.png'
        misc.imsave(websitePath + imgPath, images[node])
        s = '{"id": ' + str(node) + ', "line": ' + str(pathInd[node])
        s += ', "faces": [' + ','.join([str(x) for x in np.nonzero(faces[node])[0]]) + ']'
        s += ', "time": ' + str(years[node]) + ', "long": ' + str(longs[node]) + ', "lat": ' + str(lats[node])
        s += '}'
        strs.append(s)
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
            # be a bit careful to deal with overlapping clusters
            if j+1 in clusterInd.keys():
                clusterInd[j+1].append(i+1)
            else:
                clusterInd[j+1] = [i+1]
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


################### MAIN HELPERS ########################


def getWeights(faces, times, places):
    """
    Weight importance of faces, times, places by frequency (normalized).
    """
    # Need to run checks in case the feature vectors are length 0
    if len(faces.shape) < 2 or faces.shape[1] == 0:
        faceWeights = np.zeros(0)
    else:
        faceWeights = np.apply_along_axis(np.sum, 0, faces)
        faceWeights = faceWeights / np.linalg.norm(faceWeights)
    if len(times.shape) < 2 or times.shape[1] == 0:
        timeWeights = np.zeros(0)
    else:
        timeWeights = np.apply_along_axis(np.sum, 0, times)
        timeWeights = timeWeights / np.linalg.norm(timeWeights)
    if len(places.shape) < 2 or places.shape[1] == 0:
        placeWeights = np.zeros(0)
    else:
        placeWeights = np.apply_along_axis(np.sum, 0, places)
        placeWeights = placeWeights / np.linalg.norm(placeWeights)
    return np.hstack([faceWeights, timeWeights, placeWeights])


def getPool(photos, cl, faces, times, pairDists):
    """
    Return pool of photos containing faces in cl to initiate path construction.
    Uses times and pairDists for coherence. Bit of a hack.
    """
    # choose a pool of photos with sufficient number of people from cluster present
    nonz = np.nonzero(faces[:,cl])[0]
    sumz = dict(zip(nonz, np.apply_along_axis(np.count_nonzero, 1, faces[nonz][:,cl])))
    pool = filter(lambda x: x in sumz.keys() and sumz[x]>len(cl)*TAU, photos)
    
    # choose a starter photo that has most sufficiently coherent photos
    maxPool = []
    for i in pool:
        candidates = coherence([i], pool, pairDists, times)
        if len(candidates) > len(maxPool):
            maxPool = candidates
    return maxPool


def getLandmarkDistances(ppl, landmarks):
    """
    Get pairwise distance vectors for landmark info.
    """
    means = np.mean(landmarks, axis=2) #avg along landmarks : 988 x 223 x 2
    pairDists = np.array([means[:,ppl[i],:] - means[:,ppl[j],:] for i in range(len(ppl)) for j in range(i+1, len(ppl))])
    pairDists = np.swapaxes(pairDists, 0, 1)
    return pairDists


def fixLines(paths, faceClusters, faces, years):
    """
    Do a few things to fix paths:
    -put photos in other lines to show overlaps
    -sort by year
    -order to improve layout
    """
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
    return paths


################### MAIN METHODS ########################


def makeMap(prefix, faceClusters, faces, years, longitudes, latitudes, landmarks):
    """
    Do everything to make a map. (enables zooming)
    Prefix is a file prefix to use for the map
    faces, years, etc. are the feature matrices
    """
    n, m = faces.shape
    photos = np.arange(n)
    ppl = np.arange(m)

    # Set up some additional feature vectors and weights
    times, places = binValues(years, longitudes, latitudes)
    xs = np.hstack([faces, times, places]) #feature vectors for coverage
    weights = getWeights(faces, times, places)
    pairDists = getLandmarkDistances(list(set(cbook.flatten(faceClusters))), landmarks) #pairwise distance for coherence
    
    # For each face cluster, get high-coverage coherent path for its photos
    paths = []
    for cl in faceClusters:

        # modify weights to de-emphasize faces outside the cluster
        newWeights = np.copy(weights)
        indices = list(set(ppl) - set(cl))
        newWeights[indices] *= -1
        
        # Get pool of candidate photos
        pool = getPool(photos, cl, faces, times, pairDists)

        # Build a path via greedy choices from pool, using coherence to refine pool
        path = []
        for i in range(NUM_TIMES):
            greedy(paths, path, pool, xs, newWeights)
            pool = coherence(path, pool, pairDists, times)
        paths.append(path)

    # Post-processing to tweak lines (mostly aesthetically)
    paths = fixLines(paths, faceClusters, faces, years)
    
    return paths


def mapFromParams(prefix, facelist, timeframe, longframe, latframe):
    """
    Make a metro map from the given parameters.

    prefix: output file prefix
    facelist: list of faces (indices from contacts.xml) to include
    timeframe: tuple of times (start, end)
    longframe: tuple of longitudes (start, end)
    latframe: tuple of latitudes (start, end)
    """
    mat = io.loadmat('../data/April_full_gps.mat')
    images = mat['images'][:,0]
    faces = mat['facesBinary'] #mat['faces]
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()
    landmarks = io.loadmat('../data/landmarks.mat')['landmarks'] # 988 x 223 x 49 x 2
    landmarks = np.vstack([landmarks, np.zeros((1, 223, 49, 2))]) # wtf is wrong with this

    ########### PRE-PROCESSING ###############

    # Figure out pool of photos, using params
    pool = np.arange(faces.shape[0]) # list of indices of photos we're using
    mask = np.array([True for i in pool]) # mask for filtering dataset

    # Only include photos that include at least one of the faces in facelist
    if len(facelist) > 0:
        facelist = [x+1 for x in facelist]
        mask &= faces[:, facelist].any(axis=1)
    # Only include photos that are within the given timeframe, longframe, latframe
    if len(timeframe) > 0:
        mask &= np.ma.masked_where(np.logical_and(years > timeframe[0], years < timeframe[1]), years).mask
    if len(longframe) > 0:
        mask &= np.ma.masked_where(np.logical_and(longitudes > longframe[0], longitudes < longframe[1]), longitudes).mask
    if len(latframe) > 0:
        mask &= np.ma.masked_where(np.logical_and(latitudes > latframe[0], latitudes < latframe[1]), latitudes).mask

    pool = pool[mask]
    images = images[pool]
    faces = faces[pool]
    years = years[pool]
    longitudes = longitudes[pool]
    latitudes = latitudes[pool]
    landmarks = landmarks[pool]

    # Omit people who don't appear -- TODO not sure if we'll need this
#    names = names[~np.all(faces == 0, axis=0)]
#    faces = faces[:, ~np.all(faces == 0, axis=0)]
    n, m = faces.shape
    ppl = np.arange(m)

    # Form adjacency matrix of the social graph (including ME)
    A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in ppl for j in ppl]).reshape((m,m))
    # now omit ME
    faces = faces[:,1:]
    landmarks = landmarks[:, 1:]

    # Form hashtable of (group of faces) -> (photo count), and correct timestamps
    faceClusters = clusterFaces(faces)
    correctTimestamps(years, faces, faceClusters)
    faceClusters = faceClusters[:NUM_LINES]

    ########### MAKING THE MAP ###############

    paths = makeMap(prefix, faceClusters, faces, years, longitudes, latitudes, landmarks)

    # Save map to json
    np.place(years, np.isinf(years), np.min(years))
    np.place(longitudes, np.isinf(longitudes), 0)
    np.place(latitudes, np.isinf(latitudes), 0)
    saveMap(websitePath + prefix + '-map.json', paths, images, faces, years, longitudes, latitudes)
    # Save adjacency matrix to json
    saveGraph(websitePath + prefix + '-graph.json', A, faceClusters, names)


if __name__ == '__main__':
    mapFromParams(prefix, [], [], [], [])
