# April Shen
# 4/1/2014
# Metro maps X Photobios - Main file
# This is the version turned in for CSE 576, with full metro maps-esque machinery
#!/usr/bin/python

from __future__ import division
import sys
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from random import sample, choice, shuffle, random
import Queue
from baselines import yearsBaseline, kmedoids


# Importance weights for features
faceWeights = []
timeWeights = []
placeWeights= []

# This is index is me.  Eventually this gets incorporated into the UI...
ME = 0
# This is the minimum coherence constraint. Range: (0, 1]
TAU = 0.6
# This governs the connectivity/coverage tradeoff. Range: (0, 3]
EPSILON = 0.001


def coverage(map, faces, times, places):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = list(set(cbook.flatten(map)))
    # Compute coverages, vector-style
    vects = np.hstack([faces, times, places]) #XXX regions
    weights = np.hstack([faceWeights, timeWeights, placeWeights])
    total = (1 - np.prod(1 - vects[subset], axis=0)).dot(weights)
    return total


def coherence(chain, faces, times):
    """
    Compute the coherence of the given chain.
    Coherence is based on what faces are included and chronology.
    """
    # Coherence is min. number of faces shared between two
    # images, weighted by frequency. also penalized if time moves backwards.

    # XXX what about focus on small group of people? ('activations')
    minShare = float('inf')
    for i in np.arange(len(chain)-1):
        pic1 = chain[i]
        pic2 = chain[i+1]
        #faces shared by both
        numShare = faces[pic1].dot(faceWeights * faces[pic2])
        #if time moves backwards, penalize XXX hack alert
        if (times[pic1] != 0).any() and (times[pic2] != 0).any:
            if np.nonzero(times[pic1])[0][0] > np.nonzero(times[pic2])[0][0]:
                numShare /= 2.0
            
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


def incCover(p, M, faces, times, places):
    """
    Incremental coverage of new path p over current map M.
    """
    other = M + [p]
    return coverage(other, faces, times, places) - coverage(M, faces, times, places)


def CELF(map, candidates, faces, times, places):
    """
    Use Leskovec et al.'s CELF to choose best coverage path from
    candidates to add to map. Faster than greedy through lazy
    evaluation of incremental coverage
    """
    if candidates.empty():
        return
    while True:
        # Compute incremental coverage for top value in pq
        d, p = candidates.get()
        cov = incCover(p[0], map, faces, times, places)
        # If previously computed, we've found the best path
        if p[1]:
            map.append(p[0])
            break
        else:
            candidates.put((-cov, (p[0], True)))


def buildCoherenceGraph(faces, times, m=3, tau=2, maxIter=10):
    """
    Form coherence graph G, where each node is an m-length coherent
    path and an edge exists if the nodes overlap by m-1 steps. Every
    node has coherence at least tau. We do this using the general
    best-first search strategy employed in metro maps.
    Returns list of nodes (chains) and adjacency matrix.
    """
    items = np.arange(faces.shape[0])
    # priority queue of subchains
    pq = Queue.PriorityQueue()
    # nodes of the graph
    nodes = []

    # Fill queue with pairs with sufficiently high coherence
    for i in items:
        for j in items:
            if i != j:
                c = coherence([i,j], faces, times)
                if c > tau:
                    pq.put((-c, [i,j])) #negative because we want max coherence first

    iter = 0
    while not pq.empty() and iter < maxIter:
        # Expand the chain with highest coherence using all possible extensions
        coh, chain = pq.get()
        # shuffle to add randomness
        shuffle(items)
        for i in items:
            if i not in chain:
                newChain = chain + [i]
                c = coherence(newChain, faces, times)
                if c > tau:
                    # If we reach length m, make a new node
                    if len(newChain) >= m:
                        nodes.append(newChain)
                    else:
                        pq.put((-c, newChain))
        iter += 1

    # Add edges to the graph
    n = len(nodes)
#    edges = np.zeros((n, n))
    edges = np.eye(n) #XXX connect nodes to selves...
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            u = nodes[i]
            v = nodes[j]
            # If overlap by m-1, add an edge
            hasEdge1 = hasEdge2 = True
            for k in range(1, m):
                if u[k] != v[k-1]:
                    hasEdge1 = False
                if v[k] != u[k-1]:
                    hasEdge2 = False
            if hasEdge1:
                edges[i, j] = 1
            if hasEdge2:
                edges[j, i] = 1

    return nodes, edges


def RG(s, t, B, map, nodes, edges, bPaths, faces, times, places, maxRecur=5):
    """
    Recursive greedy algorithm to solve submodular orienteering
    problem.  Finds s-t walk of length at most B, with maximum
    recursion depth maxRecur. Uses current estimate of map (set of chains).
    Taken from (Chekuri and Pal, 2005).
    Returns the path and its incremental coverage
    """
    # If no B-length s-t path, infeasible
    if bPaths[s, t] == 0:
        return [], 0.0

    if maxRecur == 0:
        # If found a neighboring pair, this is the best path
        if edges[s, t] > 0:
            p = [nodes[s], nodes[t]]
            return p, incCover(p, map, faces, times, places)
        # Otherwise infeasible
        return [], 0.0
    P = []
    m = 0.0
    
    # Guess middle node and cost to reach it, and recurse
    # Note that the only candidate middle nodes are those within B from both s and t
    guesses = set(np.nonzero(bPaths[s])[0]) & set(np.nonzero(bPaths[:,t])[0]) #set intersect
    for v in guesses: #np.arange(len(nodes)):
        for b in np.arange(1, B+1):
            # If either of these are infeasible, try another b
            p1, c1 = RG(s, v, b, map, nodes, edges, bPaths, faces, times, places, maxRecur-1)
            if len(p1) == 0:
                continue
            p2, c2 = RG(v, t, B-b, map + [p1], nodes, edges, bPaths, faces, times, places, maxRecur-1)
            if len(p2) == 0:
                continue

            newM = incCover(p1 + p2, map, faces, times, places)
            if newM > m:
                P = p1 + p2[1:] #start at 1 to omit the shared node
                m = newM
    return P, m


def getCoherentPaths(nodes, edges, faces, times, places, l=3, k=2, maxRecur=5):
    """
    Return set of k l-coherent paths in G = (nodes, edges) that
    maximize coverage.  We accomplish this through submodular
    orienteering with recursion depth maxRecur to find maximum coverage paths
    between every two nodes in G, then we greedily choose the best
    path to add to the map and repeat.
    """
    map = []
    imgs = np.arange(faces.shape[0])
    B = l - len(nodes[0])
    # Count paths <= B length in graph
    bPaths = np.empty(edges.shape)
    np.copyto(bPaths, edges)
    for i in np.arange(1, B):
        bPaths += bPaths.dot(edges)

    # Build dict of img -> list of nodes beginning with img,
    # and another for list of nodes ending with img
    beginsWith = {}
    endsWith = {}
    for im in imgs:
        begs = []
        ends = []
        for node in nodes:
            if node[0] == im:
                begs.append(nodes.index(node))
            if node[-1] == im:
                ends.append(nodes.index(node))
        beginsWith[im] = begs
        endsWith[im] = ends

    # The top candidate is our top coverage path
#    map.append(candidates.get()[1][0])
#    print 'done with iteration 0'

    for iter in np.arange(k):
        # Candidates is a priority queue ordered in decreasing coverage
        candidates = Queue.PriorityQueue()
        conn = connectivity(map, faces)

        # Use orienteering to get list of candidate paths
        for im1 in imgs:
            starts = beginsWith[im1] #sample(beginsWith[im1], int(len(beginsWith[im1]) / 2.0))
            for im2 in imgs:
                if im1 == im2:
                    continue
                ends = endsWith[im2] #sample(endsWith[im2], int(len(endsWith[im2]) / 2.0))
                maxPath = []
                maxCov = 0.0
                maxConn = conn

                # Find best-coverage path between each pair of images
                for s in starts:
                    for t in ends:
                        p, c = RG(s, t, B, map, nodes, edges, bPaths, faces, times, places, maxRecur)
                        if (len(map) == 0 and c > maxCov) or c > maxCov + EPSILON:
                            maxPath = p
                            maxCov = c
                        elif len(map) != 0 and abs(c - maxCov) < EPSILON:
                            # If close to max coverage, only choose if greater connectivity
                            newConn = connectivity(map + [p], faces)
                            if newConn > maxConn:
                                print 'conn wins'
                                maxPath = p
                                maxCov = c
                                maxConn = newConn
                        
                # we keep 1 candidate per pair of images
                if len(maxPath) > 0:
                    print maxPath
#                    candidates.put((-maxCov, maxPath))
                    candidates.put((-maxCov, (maxPath, False))) #false is for use by celf

        # Greedily choose best candidate
        CELF(map, candidates, faces, times, places)
#        map.append(candidates.get()[1])
        print 'done with iteration', iter+1
#        if iter == k-2:
#            break

        # reset whether computed or not
 #       other = Queue.PriorityQueue()
 #       while not candidates.empty():
 #           d, p = candidates.get()
 #           other.put((d, (p[0], False)))
 #       candidates = other

    # Flatten paths and return map
    # (I wrote this all out to make it super explicit and less confusing)
        ## XXX can probably flatten earlier now
    M = []
    for p in map:
        newP = []
        for node in p:
            for img in node:
                if img not in newP:
                    newP.append(img)
        M.append(newP)
    return M


def getConnections(map, faces):
    """
    Return a list of pairwise connections, where two photos are
    connected if they are in different lines and share at least one
    face.
    """
    connections = []
    numLines = len(map)

    for u in np.arange(numLines):
        for v in np.arange(u+1, numLines):
            for i in map[u]:
                for j in map[v]:
                    if faces[i].dot(faces[j]) > 0:
                        connections.append([i,j])
    return connections


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
        k = int(args[1]) # number of year bins to select
        # I imagine this being some function of the UI, so zooming can change this

    # Load data
    mat = io.loadmat('../data/April_full_new.mat') #'../data/April_full_fixedTime_binary.mat')
    images = mat['images'][:,0]
    n = images.shape[0]
    items = np.arange(n)

    years = mat['timestamps'].reshape(n)
    longitudes = mat['longitudes'].reshape(n)
    latitudes = mat['latitudes'].reshape(n)
    faces = mat['faces']
    facesBin = mat['facesBinary']
    faces += facesBin
    print 'done loading'

    # FOR TESTING XXX
#     choices = sample(items, 500)
#     images = images[choices]
#     years = years[choices]
#     longitudes = longitudes[choices]
#     latitudes = latitudes[choices]
#     faces = faces[choices]
#     n = images.shape[0]
#     items = np.arange(n)

    # Bin times and GPS coordinates. If value is missing we assume it covers nothing.
    times = bin(years, k)
    longs = bin(longitudes, k)
    lats = bin(latitudes, k)

    # just need place-bins
    places = np.zeros((n, k**2))
    nonLongs = np.nonzero(longs)
    nonLats = np.nonzero(lats)
    for img, lo, la in zip(nonLongs[0], nonLongs[1], nonLats[1]):
        places[img, lo + la*k] = 1
    places = places[:, ~np.all(places == 0, axis=0)]

    # Get rid of ME
    faces = np.hstack([faces[:,:ME], faces[:,ME+1:]])

    # Weight importance of faces, times, places by frequency... normalize
    faceWeights = np.apply_along_axis(np.count_nonzero, 0, faces)
    faceWeights = faceWeights / np.linalg.norm(faceWeights)
    timeWeights = np.apply_along_axis(np.count_nonzero, 0, times)
    timeWeights = timeWeights / np.linalg.norm(timeWeights)
    placeWeights = np.apply_along_axis(np.count_nonzero, 0, places)
    placeWeights = placeWeights / np.linalg.norm(placeWeights)

    # Find high-coverage coherent paths
    nodes, edges = buildCoherenceGraph(faces, times, m=3, tau=TAU, maxIter=200) #pretty fast
    print 'number of nodes', len(nodes)
    print 'done building graph'
    paths = getCoherentPaths(nodes, edges, faces, times, places, l=5, k=5, maxRecur=2) #sure as hell not fast
    print 'done getting paths'

    # Get connections between lines
    connections = getConnections(paths, faces)

    # Save map to csv
    # Each image is separated by a comma, each path by a linebreak
    output = open('large_run.csv', 'w+')
    
    # First include connections
    output.write(','.join(map(str, list(cbook.flatten(connections)))) + '\n')
    for p in paths:
        output.write(','.join(map(str, p)) + '\n')
    output.close()

    # Display paths
    for i in range(len(paths)):
        plt.figure(i)
        path = paths[i]
        for j, img in zip(range(len(path)), path):
            plt.subplot(1, len(path), j+1)
            plt.title('image ' + str(img))
            plt.imshow(images[img])
    plt.show()
