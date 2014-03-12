# April Shen
# Metro maps X Photobios - Main file
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


def coverage(map, faces, times, places):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = set(cbook.flatten(map))
    total = 0.0
    numImages, numPeople = faces.shape
    numTimes = times.shape[0]
    numPlaces = places.shape[0]
    imgs = np.arange(numImages)
    ppl = np.arange(numPeople)
    tim = np.arange(numTimes)
    plac = np.arange(numPlaces)

    # Weight importance of faces, times, places by frequency...
    faceWeights = np.apply_along_axis(np.count_nonzero, 1, faces)
    timeWeights = np.apply_along_axis(np.count_nonzero, 0, times)
    placeWeights = np.apply_along_axis(np.count_nonzero, 0, places)

    # Compute coverages
    for v, w in zip(ppl, faceWeights):
        c = 1.0
        for u in subset:
            c *= (1 - faces[u,v]) #includes size of region
        c = 1 - c
        total += w * c
    for v, w in zip(tim, timeWeights):
        c = 1.0
        for u in subset:
            c *= (1 - times[u,v])
        c = 1 - c
        total += w * c
    for v, w in zip(plac, placeWeights):
        c = 1.0
        for u in subset:
            c *= (1 - places[u,v])
        c = 1 - c
        total += w * c

    return total


def coherence(chain, faces, times):
    """
    Compute the coherence of the given chain.
    Coherence is based on what faces are included and chronology.
    """
    # FOR NOW, coherence is min. number of faces shared between two
    # images. also penalized if time moves backwards.

    # XXX what about focus on small group of people? ('activations')
    # maybe the regions weighting will account for this in a way...
    minShare = float('inf')
    # Weight faces by frequency
    #faceWeights = np.apply_along_axis(np.count_nonzero, 1, faces)

    for i in np.arange(len(chain)-1):
        pic1 = chain[i]
        pic2 = chain[i+1]
        #faces shared by both
        numShare = faces[pic1].dot(faces[pic2])
        #if time moves backwards, subtract something XXX hack alert
        if (times[pic1] != 0).any() and (times[pic2] != 0).any:
            if np.nonzero(times[pic1])[0][0] > np.nonzero(times[pic2])[0][0]:
                numShare /= 2.0
            
        if numShare < minShare:
            minShare = numShare
    return minShare


def connectivity(map, faces):
    """
    Compute connectivity of the map. Two lines are considered
    connected if their nodes share faces (and maybe times?)
    """
    if len(map) < 2:
        return 0
    # Try counting number of lines that intersect
    numLines = len(map)
    flatMap = []
    for u in np.arange(numLines):
        flatMap.append(set(cbook.flatten(map[u])))
    total = 0
    for u in np.arange(numLines):
        for v in np.arange(u+1, numLines):
            intersect = False
            for i in flatMap[u]:
                for j in flatMap[v]:
                    if faces[i].dot(faces[j]) > 0:
                        intersect = True
                        break
                if intersect:
                    break
            if intersect:
                total += 1
#            total += np.sum([faces[i].dot(faces[j]) for i in flatMap[u] for j in flatMap[v]])
    return total


def incCover(p, M, faces, times, places):
    """
    Incremental coverage of new path p over current map M.
    """
    other = M + [p]
    return coverage(other, faces, times, places) - coverage(M, faces, times, places)


def greedy(map, candidates, faces, times):
    """
    Greedily choose the best path from candidates (based on maximizing
    coverage) and add to map.
    """
    maxCoverage = 0.0
    maxPath = 0
    for p in candidates:
        # Find the path that adds the most coverage
        c = coverage(map + [p], faces, times)
        if c > maxCoverage:
            maxCoverage = c
            maxPath = p
    map.append(maxPath)


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


def RG(s, t, B, map, nodes, edges, bPaths, faces, times, places, i=5):
    """
    Recursive greedy algorithm to solve submodular orienteering
    problem.  Finds s-t walk of length at most B, with maximum
    recursion depth i. Uses current estimate of map (set of chains).
    Taken from (Chekuri and Pal, 2005).
    Returns the path and its incremental coverage
    """
    # If no B-length s-t path, infeasible
    if bPaths[s, t] == 0:
        return [], 0.0

    if i == 0:
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
            p1, c1 = RG(s, v, b, map, nodes, edges, bPaths, faces, times, places, i-1)
            if len(p1) == 0:
                continue
            p2, c2 = RG(v, t, B-b, map + [p1], nodes, edges, bPaths, faces, times, places, i-1)
            if len(p2) == 0:
                continue

            newM = incCover(p1 + p2, map, faces, times, places)
            if newM > m:
                P = p1 + p2[1:] #start at 1 to omit the shared node
                m = newM
    return P, m


def getCoherentPaths(nodes, edges, faces, times, places, l=3, k=2, i=5):
    """
    Return set of k l-coherent paths in G = (nodes, edges) that
    maximize coverage.  We accomplish this through submodular
    orienteering with recursion depth i to find maximum coverage paths
    between every two nodes in G, then we greedily choose the best
    path to add to the map and repeat.
    """
    # TODO: kill the duplicate nodes dude
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

    # Candidates is a priority queue ordered in decreasing coverage
    candidates = Queue.PriorityQueue()

    # Use orienteering to get list of candidate paths first
    for im1 in imgs:
        starts = sample(beginsWith[im1], int(len(beginsWith[im1]) / 2.0))
        for im2 in imgs:
            ends = sample(endsWith[im2], int(len(endsWith[im2]) / 2.0))
            maxPath = []
            maxCov = 0.0

            # Find best-coverage path between each pair of images
            for s in starts:
                for t in ends:
                    p, c = RG(s, t, B, map, nodes, edges, bPaths, faces, times, places, i)
                    if c > maxCov:
                        maxPath = p
                        maxCov = c
                        
            # we keep 1 candidate per pair of images
            if len(maxPath) > 0:
                print maxPath
                candidates.put((-maxCov, (maxPath, False))) #false is for use by celf

    # The top candidate is our top coverage path
    map.append(candidates.get()[1][0])

    # Find best-coverage paths using CELF
    # XXX ignore connectivity for now
    for iter in np.arange(k-1):
        CELF(map, candidates, faces, times, places)
        print 'done with iteration', iter
        if iter == k-2:
            break

        # reset whether computed or not
        other = Queue.PriorityQueue()
        while not candidates.empty():
            d, p = candidates.get()
            other.put((d, (p[0], False)))
        candidates = other

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

    # TODO filter out me
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
    valid = items[np.isfinite(values)]
    sortedVals = sorted(valid, key=lambda i: values[i])
    num = int(np.ceil(len(sortedVals) / k))
    bins = [sortedVals[x:x+num] for x in range(0, len(sortedVals), num)]
    vectors = np.zeros((n, len(bins)))
    for i in items:
        if np.isfinite(values[i]):
            whichBin = map(lambda x: i in x, bins).index(True)
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
    faces = mat['facesBinary'] #xxx
    t = 3 ##0.5 # note when we change btw binary and not, need to change tau XXX
    print 'done loading'

    # FOR TESTING XXX
    choices = sample(np.arange(images.shape[0]), 500)
    images = images[choices]
    years = years[choices]
    longitudes = longitudes[choices]
    latitudes = latitudes[choices]
    faces = faces[choices]

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

    # Find high-coverage coherent paths
    nodes, edges = buildCoherenceGraph(faces, times, m=3, tau=t, maxIter=100) #pretty fast
    print 'number of nodes', len(nodes)
    print 'done building graph'
    paths = getCoherentPaths(nodes, edges, faces, times, places, l=5, k=4, i=2) #sure as hell not fast
    print 'done getting paths'

    # Get connections between lines
    connections = getConnections(paths, faces)

    # Save map to csv
    # Each image is separated by a comma, each path by a linebreak
    output = open('fasterBiggerToo.csv', 'w+')
    
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
