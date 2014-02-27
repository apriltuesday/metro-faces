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


def robustLogistic(x):
    return pow(1 + np.exp(-np.log(99) * x), -1)


def coverage(map, faces):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = set(cbook.flatten(map))
    total = 0.0
    numImages, numPeople = faces.shape
    imgs = np.arange(numImages)
    ppl = np.arange(numPeople)
    # Weight importance by frequency... TODO time, importance, size of region....
    weights = np.apply_along_axis(np.count_nonzero, 1, faces)

    for v, w in zip(ppl, weights):
        c = 1.0
        for u in subset:
            c *= (1 - faces[u,v])
        c = 1 - c
        total += w * c
    return total


def coherence(chain, faces, times):
    """
    Compute the coherence of the given chain.
    Coherence is based on what faces are included and chronology.
    """
    # FOR NOW, coherence is min. number of faces shared between two
    # images, no time included yet. (TODO)
    minShare = float('inf')
    for i in np.arange(len(chain)-1):
        numShare = faces[chain[i]].dot(faces[chain[i+1]])
        if numShare < minShare:
            minShare = numShare
    return minShare


def incCover(p, M, faces):
    """
    Incremental coverage of new path p over current map M.
    """
    other = M + [p]
    return coverage(other, faces) - coverage(M, faces)


def greedy(map, candidates, faces):
    """
    Greedily choose the best path from candidates (based on maximizing
    coverage) and add to map.
    """
    maxCoverage = 0.0
    maxPath = 0
    for p in candidates:
        # Find the path that adds the most coverage
        c = coverage(map + [p], faces)
        if c > maxCoverage:
            maxCoverage = c
            maxPath = p
    map.append(maxPath)


def CELF(map, candidates, faces):
    """
    Use Leskovec et al.'s CELF to choose best coverage path from
    candidates to add to map. Faster than greedy through lazy
    evaluation of incremental coverage
    """
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
        cov = incCover(p, map, faces)
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
#    edges = np.eye(n) #need connectivity from node to self...
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


def RG(s, t, B, map, nodes, edges, bPaths, faces, i=5):
    """
    Recursive greedy algorithm to solve submodular orienteering
    problem.  Finds s-t walk of length at most B, with maximum
    recursion depth i. Uses current estimate of map (set of chains).
    Taken from (Chekuri and Pal, 2005).
    """
    # If no B-length s-t path, infeasible
    if bPaths[s, t] == 0:
        return []
    #XXX why is the overlap still sometimes wrong?
    #XXX missing middle nodes????

    if i == 0:
        # If found a neighboring pair, this is the best path
        if edges[s, t] > 0:
            return [nodes[s], nodes[t]]
        # Otherwise infeasible
        return []
    P = []
    m = 0.0#incCover(P, map, faces)
    
    # Guess middle node and cost to reach it, and recurse
    for v in np.arange(len(nodes)):
        for b in range(1, B+1):
            # If either of these are infeasible, get out now
            p1 = RG(s, v, b, map, nodes, edges, bPaths, faces, i-1)
            if len(p1) == 0:
                continue
            p2 = RG(v, t, B-b, map + [p1], nodes, edges, bPaths, faces, i-1)
            if len(p2) == 0:
                continue
#            print 'p1', p1
#            print 'p2', p2
            newM = incCover(p1 + p2, map, faces)
            if newM > m:
                P = p1 + p2[1:] #start at 1 to omit the shared node
                m = newM
    return P


def getCoherentPaths(nodes, edges, faces, l=3, k=2, i=5):
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

    # Find best paths between pairs of images
    for iter in np.arange(k):
        candidates = []
        # For each pair of images, get the nodes that start/end with them
        for im1 in imgs: #how to speed up these loops? XXX
            starts = beginsWith[im1]
            for im2 in imgs:
                ends = endsWith[im2]
                maxPath = []

# XXX also could flatten paths earlier...

            # Then find best-coverage path between each of these nodes
                for s in starts:
                    for t in ends:
                        p = RG(s, t, B, map, nodes, edges, bPaths, faces, i)
                        if len(p) > len(maxPath): #found a better candidate path
                            maxPath = p
                            print p

                # Save the best of these paths between the two images
                if len(maxPath) > 0:
                    print 'best', maxPath
                    candidates.append(maxPath)
        
        # Greedily choose best candidate and add to map -- using CELF!
        CELF(map, candidates, faces)
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


if __name__ == '__main__':
#    args = sys.argv
#    if len(args) < 2:
#        k = 5
#    else:
#        k = int(args[1]) # number of images to select

    # Load data
    mat = io.loadmat('../data/April_full_dataset_binary.mat')
    images = mat['images'][:,0]
    times = mat['timestamps']
    faces = mat['faces']
    items = np.arange(len(images))
    print 'done loading'

    # Find high-coverage coherent paths
    nodes, edges = buildCoherenceGraph(faces, times, m=3, tau=3, maxIter=100) #pretty fast
    print 'done building graph'
    paths = getCoherentPaths(nodes, edges, faces, l=5, k=3, i=5) #sure as hell not fast
    print 'done getting paths'

    # Display paths
    for i in range(len(paths)):
        plt.figure(i)
        path = paths[i]
        for j, img in zip(range(len(path)), path):
            plt.subplot(1, len(path), j+1)
            plt.title('image ' + str(img))
            plt.imshow(images[img])
    plt.show()
