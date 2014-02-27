# April Shen
# Metro maps X Photobios - Trying out coherence
#!/usr/bin/python

from __future__ import division
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import Queue
from random import shuffle


def coherence(chain, faces, times):
    """
    Compute the coherence of the given chain.
    Coherence is based on what faces are included and chronology.
    """
    # FOR NOW, coherence is min. number of faces shared between two
    # images, no time included yet.
    minShare = float('inf')
    for i in np.arange(len(chain)-1):
        numShare = faces[chain[i]].dot(faces[chain[i+1]])
        if numShare < minShare:
            minShare = numShare
    return minShare


def buildCoherenceGraph(faces, times, m, tau, maxIter):
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


def incCover(p, M):
    """
    Incremental coverage of p over M.
    """
    other = M.append(p)
    return coverage(other) - coverage(M)


def RG(s, t, B, X, i, map):
    """
    Recursive greedy algorithm to solve submodular orienteering problem.
    Finds s-t walk of length at most B finding path augmenting X, with
    maximum recursion depth i. Uses current map (set of chains).
    """
    P = [s, t]
    if i == 0:
        return P
    m = incCover(P, map)
    for v in nodes:
        for b in range(1, B+1):
            p1 = RG(s, v, b, X, i-1, map)
            p2 = RG(v, t, B-b, X + P, i-1, map)
            newM = incCover(p1 + p2, map)
            if newM > m:
                P = p1 + p2
                m = newM
    return P


def getCoherentPaths(G, l, k):
    """
    Return set of k l-coherent paths in G that maximize coverage.  We
    accomplish this through submodular orienteering to find maximum
    coverage paths between every two nodes in G, then we greedily
    choose the best path to add to the map and repeat.
    """
    pass


if __name__ == '__main__':
    mat = io.loadmat('../data/April_full_dataset.mat')
    images = mat['images'][:,0]
    times = mat['timestamps']
    faces = mat['faces']

    nodes, edges = buildCoherenceGraph(faces, times, 3, 0.1, 10)
    xs,ys = np.nonzero(edges)
    print nodes[xs[0]]
    print nodes[ys[0]]
