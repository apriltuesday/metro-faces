# April Shen
# Metro maps X Photobios - Main file
#!/usr/bin/python

from __future__ import division
import sys
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from random import sample, choice, shuffle
import Queue
from baselines import yearsBaseline, kmedoids


def coverage(map, faces, times):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = set(cbook.flatten(map))
    total = 0.0
    numImages, numPeople = faces.shape
    numTimes = times.shape[1]
    imgs = np.arange(numImages)
    ppl = np.arange(numPeople)
    tim = np.arange(numTimes)

    # Weight importance of faces by frequency...
    faceWeights = np.apply_along_axis(np.count_nonzero, 1, faces)
    # Weight importance of times by frequency (for now... maybe recency?) XXX
    timeWeights = np.apply_along_axis(np.count_nonzero, 0, times)
    timeWeights = timeWeights / np.linalg.norm(timeWeights) #normalize... necessary?

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
        #if time moves backwards, subtract something
        if np.nonzero(times[pic1])[0] > np.nonzero(times[pic2])[0]:
            numShare -= 1
            
        if numShare < minShare:
            minShare = numShare
    return minShare


def connectivity(map, faces):
    """
    Compute connectivity of the map. Two lines are considered
    connected if their nodes share faces (and maybe times?)
    """
    # Try counting number of lines that intersect
    numLines = len(map)
    flatMap = []
    for u in np.arange(numLines):
        flatMap.append(set(cbook.flatten(map[u])))
    total = 0
    for u in np.arange(numLines):
        for v in np.arange(u+1, numLines):
            intersect = False
            # XXX times?
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


def incCover(p, M, faces, times):
    """
    Incremental coverage of new path p over current map M.
    """
    other = M + [p]
    return coverage(other, faces, times) - coverage(M, faces, times)


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


def CELF(map, candidates, faces, times):
    """
    Use Leskovec et al.'s CELF to choose best coverage path from
    candidates to add to map. Faster than greedy through lazy
    evaluation of incremental coverage
    """
    if len(candidates) == 0:
        return
    # priority queue sorted by incremental coverage
    pq = Queue.PriorityQueue()
    for p in candidates:
        pq.put((-float('inf'), p)) #negative because want max first
    # array indicating whether we've computed coverage for a path before
    computed = [False for p in candidates]

    while True:
        # Compute incremental coverage for top value in pq
        d, p = pq.get()
        i = candidates.index(p)
        cov = incCover(p, map, faces, times)
        # If previously computed, we've found the best path
        if computed[i]:
            map.append(p)
            break
        else:
            pq.put((-cov, p))
            computed[i] = True


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
    edges = np.zeros((n, n))
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


def RG(s, t, B, map, nodes, edges, bPaths, faces, times, i=5):
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
            return p, incCover(p, map, faces, times)
        # Otherwise infeasible
        return [], 0.0
    P = []
    m = 0.0
    
    # Guess middle node and cost to reach it, and recurse
    for v in np.arange(len(nodes)):
        for b in range(1, B+1):
            # If either of these are infeasible, get out now
            p1, c1 = RG(s, v, b, map, nodes, edges, bPaths, faces, times, i-1)
            if len(p1) == 0:
                continue
            p2, c2 = RG(v, t, B-b, map + [p1], nodes, edges, bPaths, faces, times, i-1)
            if len(p2) == 0:
                continue
            newM = incCover(p1 + p2, map, faces, times)
            if newM > m:
                P = p1 + p2[1:] #start at 1 to omit the shared node
                m = newM
    return P, m


def getCoherentPaths(nodes, edges, faces, times, l=3, k=2, i=5):
    """
    Return set of k l-coherent paths in G = (nodes, edges) that
    maximize coverage.  We accomplish this through submodular
    orienteering with recursion depth i to find maximum coverage paths
    between every two nodes in G, then we greedily choose the best
    path to add to the map and repeat.
    """
    map = []
    imgs = np.arange(faces.shape[0])
    B = l - len(nodes[0])# + 1
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

    conn = connectivity(map, faces) #connectivity of the current map
        
    # Find best paths between pairs of images
    for iter in np.arange(k):
        candidates = []
        # For each pair of images, get the nodes that start/end with them
        for im1 in imgs: #how to speed up these loops? XXX
            starts = beginsWith[im1]
            for im2 in imgs:
                ends = endsWith[im2]
                maxPath = []
                maxCov = 0.0
                maxConn = conn

            # Then find best-coverage path between each of these nodes
                for s in starts:
                    for t in ends:
                        p, c = RG(s, t, B, map, nodes, edges, bPaths, faces, times, i)
                        # If close to max coverage, only choose if greater connectivity
                        if c > maxCov + 0.0001:
                            maxPath = p
                            maxCov = c
                            maxConn = connectivity(map + [p], faces)
                        elif abs(c - maxCov) < 0.0001:
                            newConn = connectivity(map + [p], faces)
                            if newConn > maxConn:
                                maxPath = p
                                maxCov = c
                                maxConn = newConn

#                         if len(p) > len(maxPath): #found a better candidate path
#                             maxPath = p

                # Save the best of these paths between the two images
                if len(maxPath) > 0:
                    print maxPath
                    candidates.append(maxPath)
        
        # Greedily choose best candidate and add to map -- using CELF!
        CELF(map, candidates, faces, times)
        print 'done with iteration', iter

    # Flatten paths and return map
    # (I wrote this all out to make it super explicit and less confusing)
    M = []
    for p in map:
        newP = []
        for node in p:
            for img in node:
                if img not in newP:
                    newP.append(img)
        M.append(newP)
    return M


def increaseConnectivity(map, nodes, edges, faces, times, maxIter=1):
    """
    Increase connectivity of the existing map as follows. We attempt
    to replace each line with an alternative that does not decrease
    map coverage and increases connectivity, reusing the orienteering
    algorithm to find alternatives.
    """
    # XXX Isn't this whole deal gonna be reeeeaaalllly slowwww???
    k = len(map)
    l = len(map[0])
    #copied from getCoherentPaths...
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

    for iter in np.arange(maxIter):
        cov = coverage(map, faces, times)
        for i in np.arange(k):
            # Consider current map without one path
            newMap = map[:i] + map[i+1:]

            #copied from getCoherentPaths...
            candidates = []
            # For each pair of images, get the nodes that start/end with them
            for im1 in imgs: #how to speed up these loops? XXX
                starts = beginsWith[im1]
                for im2 in imgs:
                    ends = endsWith[im2]
                    maxPath = []

                # Then find best-coverage path between each of these nodes
                    for s in starts:
                        for t in ends:
                            p = RG(s, t, B, newMap, nodes, edges, bPaths, faces, times, i=3)
                            if len(p) > len(maxPath): #found a better candidate path
                                maxPath = p

                    # Save the best of these paths between the two images
                    if len(maxPath) > 0:
                        print maxPath
                        candidates.append(maxPath)
            # Pick candidate with best connectivity among those with same coverage
            bestConn = connectivity(newMap, faces)
            bestCand = []
            for c in candidates:
                newCov = coverage(newMap + [c], faces, times)
                if abs(newCov - cov) < 0.00001:
                    newConn = connectivity(newMap + [c], faces)
                    if newConn > bestConn:
                        bestConn = newConn
                        bestCand = c

            newMap.append(bestCand)
        # If we've converged to something, stop
        if newMap == map:
            break
        map = newMap


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        k = 5
    else:
        k = int(args[1]) # number of year bins to select
        # I imagine this being some function of the UI, so zooming can change this

    # Load data
    mat = io.loadmat('../data/April_full_dataset_binary.mat')
    # note when we change btw binary and not, need to change tau XXX
    images = mat['images'][:,0]
    years = mat['timestamps']
    faces = mat['faces']
    print 'done loading'

    # FOR TESTING XXX
    choices = sample(np.arange(images.shape[0]), 300)
    images = images[choices]
    years = years[choices]
    faces = faces[choices]

    n = images.shape[0]
    items = np.arange(n)

    # Bin the times and make binary vector for each image
    # XXX do some hacks to deal with missing times - if missing, assume it covers nothing
    nonzeroYears = items[np.nonzero(years)[0]]
    sortedYears = sorted(nonzeroYears, key=lambda i: years[i])
    num = int(len(sortedYears) / k) + 1
    bins = [sortedYears[x:x+num] for x in range(0, len(sortedYears), num)]
    times = np.zeros((n, len(bins)))
    for i in items:
        if years[i] != 0:
            whichBin = map(lambda x: i in x, bins).index(True)
            times[i, whichBin] = 1

    # Find high-coverage coherent paths
    nodes, edges = buildCoherenceGraph(faces, times, m=3, tau=3, maxIter=100) #pretty fast
    print 'done building graph'
    paths = getCoherentPaths(nodes, edges, faces, times, l=5, k=2, i=2) #sure as hell not fast
    print 'done getting paths'

    # Improve connectivity
    # XXX This is hella slow, can we roll it into the coherent path alg?
#    increaseConnectivity(paths, nodes, edges, faces, times)
#    print 'done increasing connections'

    # Save map to csv
    # Each image is separated by a comma, each path by a linebreak
    output = open('largeTest.csv', 'w+')
    # TODO line 0 must be connections
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
