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
NUM_LINES = 20
TAU = 0.5 # This is the minimum coherence constraint

# Numbers of bins
NUM_CLUSTERS = 200
NUM_TIMES = 50
NUM_LOCS = 200

# For output files etc.
websitePath = '../apriltuesday.github.io/'
prefix = '5-31-14_test'


################### CORE ALGORITHM ########################


def coverage(paths, xs, weights):
    """
    Computes coverage of map, using xs as features weighted by weights.
    """
    subset = list(set(cbook.flatten(paths)))
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def coherence(path, xs, times):
    """
    Return coherence of path, based on features xs.
    times forces unique photo per time bin per path (and eventually, ordering)
    """
    maxDist = 0.0 # for now, min coherence == max feature distance
    masked = np.ma.masked_where(~np.isfinite(xs), xs)
    for p1 in path:
        for p2 in path:
            # TODO: what exactly do we want here? norm? dot prod?
            dist = np.sqrt(np.sum((masked[p1] - masked[p2])**2))
            if dist > maxDist:
                maxDist = dist
    return -maxDist


def greedy(paths, candidates, xs, weights):
    """
    Greedily choose the candidate with max coverage relative to map+path,
    and add to path. Uses xs as features with the given weights.
    """
    if len(candidates) == 0:
        return None, None
    maxCoverage = -float('inf') #note that if weights are negative, we can have negative coverage!
    totalMap = list(cbook.flatten(paths.keys())) #values()))
    for cl, p in candidates.items():
        c = coverage(totalMap + list(cl), xs, weights) #p
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
            maxCl = cl
    return maxCl, maxP


def improveStructure(paths, candidates, faces):
    """
    Local search to improve structure of paths.
    -> if two lines overlap by >50%, merge
    -> if a line increases connectivity, add
    """
    # If two disjoint lines have a common face, connect them!
    for cl1, p1 in paths.items():
        for cl2, p2 in paths.items():
            if len(set(p1) & set(p2)) == 0:
                faces1 = faces[p1]
                faces2 = faces[p2]
                common = set(np.nonzero(faces1)[1]) & set(np.nonzero(faces2)[1])
                if len(common) == 0:
                    continue
                #for cl in candidates.keys():
                #    if set(cl) < common:
                #        paths[cl] = candidates[cl]
                #        candidates.pop(cl)
                #        break
                for f in common:
                    newPath = set(np.array(p1)[np.nonzero(faces1[:,f])[0]]) | set(np.array(p2)[np.nonzero(faces2[:,f])[0]])
                    paths[f] = list(newPath)
                    break

    """
    # Add lines that don't overlap too much but increase connectivity
    for p1 in candidates:
        for p2 in paths:
            intersection = set(p1) & set(p2)
            if len(intersection) >= 1 and (len(intersection) < len(p1) / 2 and len(intersection) < len(p1) / 2):
                #toAdd.append(p1)
                paths.append(p1)
                break
    #paths += toAdd

    # Merge lines that overlap too much
    newPaths = [set(paths[0])]
    for p1 in paths[1:]:
        for p2 in newPaths:
            intersection = set(p1) & p2
            if len(intersection) > len(p1) / 2 and len(intersection) > len(p2) / 2:
                p2 |= set(p1)
                break
        newPaths.append(set(p1))
    paths = [list(x) for x in newPaths]
    """
    return paths #newPaths


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


def orderLines(paths):
    """
    Order lines and clusters according to shared images (in place).
    """
    # This makes the visualization easier and is kind of a huge hack
    i = 0
    n = len(paths)
    cls = paths.keys()
    ps = [paths[c] for c in cls]
    while i < n:
        maxInt = 0
        maxJ = 0
        for j in range(i+1, n):
            intersect = len(set(ps[i]) & set(ps[j]))
            if intersect > maxInt: #largest intersection
                maxJ = j
                maxInt = intersect
        if maxInt > 0: #swap
            temp = ps[maxJ]
            ps[maxJ] = ps[i+1]
            ps[i+1] = temp
            # also swap corresponding face clusters
            temp = cls[maxJ]
            cls[maxJ] = cls[i+1]
            cls[i+1] = temp
        i += 1
    paths = dict(zip(cls, ps))


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


def saveMap(filename, pathsDict, images, faces, years, places):
    """
    Save map in a JSON file. Also save the corresponding photos.
    Store feature data as attributes of each node.
    """
    paths = [pathsDict[cl] for cl in pathsDict.keys()]
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
        #misc.imsave(websitePath + imgPath, images[node]) #XXX suspect don't need this anymore
        s = '{"id": ' + str(node) + ', "line": ' + str(pathInd[node])
        s += ', "faces": [' + ','.join([str(x) for x in np.nonzero(faces[node])[0]]) + ']'
        p = np.nonzero(places[node])[0]
        s += ', "time": ' + str(years[node]) + ', "place": ' + str(p[0] if len(p) > 0 else -1)
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


def getLandmarkDistances(ppl, landmarks):
    """
    Get pairwise distance vectors for landmark info.
    """
    means = np.mean(landmarks, axis=2) #avg along landmarks : 988 x 223 x 2
    pairDists = np.array([means[:,ppl[i],:] - means[:,ppl[j],:] for i in range(len(ppl)) for j in range(i+1, len(ppl))])
    pairDists = np.swapaxes(pairDists, 0, 1)
    pairDists = np.array([x.flatten() for x in pairDists])
    return pairDists


def fixLines(paths, years):
    """
    Do a few things to fix paths:
    -put photos in other lines to show overlaps
    -sort by year
    -order to improve layout
    """
    # Sort and re-order lines to improve layout
    for cl in paths.keys():
        paths[cl] = sorted(paths[cl], key=lambda x: years[x])
#    orderLines(paths)
    return paths


def clusterFaces(A, faces):
    """
    Cluster faces using co-clustering of co-occurrence matrix A.
    Return the clusters we use for lines, which maximally cover the photos.
    """
    c = cluster.bicluster.SpectralCoclustering(n_clusters=NUM_CLUSTERS, svd_method='arpack')
    c.fit(A)
    clusters = [] #list of lists, each of which lists face indices in that cluster
    for i in range(NUM_CLUSTERS):
        clusters.append(c.get_indices(i)[0])        
    whichClusters = []
    for i in range(NUM_LINES):
        greedy(whichClusters, clusters, faces.T, np.ones(n))
    return whichClusters


################### MAIN METHODS ########################


def makeMap(prefix, faceClusters, faces, years, longitudes, latitudes, landmarks, poses):
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
    xs = np.hstack([faces, times, places])
    weights = getWeights(faces, times, places)

    # Get candidate lines, one per cluster
    candidates = {} # dict from cluster to candidate
    for cl in faceClusters: #[:NUM_LINES]: #XXX
        # Get pool of photos with sufficient number of ppl from cluster
        nonz = np.nonzero(faces[:,cl])[0]
        sumz = dict(zip(nonz, np.apply_along_axis(np.count_nonzero, 1, faces[nonz][:,cl])))
        pool = filter(lambda x: x in sumz.keys() and sumz[x]>=TAU*len(cl), photos)
        candidates[cl] = pool
#    paths = candidates #XXX

    # Choose candidates to optimize coverage and connectivity (future: coherence?!)
    paths = {} # dict from cluster to path
    for i in np.arange(NUM_LINES):
        cl, p = greedy(paths, candidates, faces.T, np.ones(n)) #xs, weights)
        if not p: # no more candidates
            break
        print cl
        paths[cl] = sorted(p, key=lambda x: years[x])
        candidates.pop(cl)

        # Increase weights of faces in photos in the map not yet covered by a line
        #facesInPhotos = set(np.nonzero(faces[list(cbook.flatten(paths.values()))])[1])
        #facesInClusters = set(cbook.flatten(paths.keys()))
        #indices = list(facesInPhotos - facesInClusters)
        #newWeights = weights.copy()
        #newWeights[indices] *= 2 #xxx

    # TODO: structure? adding/merging lines?

    return paths, places


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
    faces = mat['facesBinary'] + mat['faces']
    years = mat['timestamps'].flatten()
    longitudes = mat['longitudes'].flatten()
    latitudes = mat['latitudes'].flatten()
    names = getNames()
    mat = io.loadmat('../data/landmarks.mat')
    landmarks = mat['landmarks'] # 988 x 223 x 49 x 2
    poses = mat['poses'] #988 x 223 x 3
    landmarks = np.vstack([landmarks, np.zeros((1, 223, 49, 2))]) # wtf is wrong with this
    poses = np.vstack([poses, np.zeros((1, 223, 3))])

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
    poses = poses[pool]

    n, m = faces.shape
    ppl = np.arange(m)

    # Form adjacency matrix of the social graph (including ME)
    A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in ppl for j in ppl]).reshape((m,m))
    # now omit ME
    faces = faces[:,1:]
    landmarks = landmarks[:, 1:]
    poses = poses[:, 1:]

    # Form hashtable of (group of faces) -> (photo count), and correct timestamps
    faceClusters = clusterFaces(faces)
    correctTimestamps(years, faces, faceClusters)
    faceClusters = faceClusters[:NUM_CLUSTERS]

    ########### MAKING THE MAP ###############

    paths, places = makeMap(prefix, faceClusters, faces, years, longitudes, latitudes, landmarks, poses)

    # Save map to json
    np.place(years, np.isinf(years), np.mean(years))
    print 'number of paths', len(paths)
    saveMap(websitePath + prefix + '-map.json', paths, images, faces, years, places)
    # Save adjacency matrix to json
    saveGraph(websitePath + prefix + '-graph.json', A, faceClusters, names)


if __name__ == '__main__':
    mapFromParams(prefix, [], [], [], [])
