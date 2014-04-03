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
NUM_PHOTOS = 8

# Numbers of bins
NUM_CLUSTERS = 150
NUM_TIMES = 100
NUM_LOCS = 50

# Parameters of the algorithm
M = 3 # number of images in each node of coherence graph
TAU = 0.2 # This is the minimum coherence constraint.
EPSILON = 0.001 # This governs the connectivity/coverage tradeoff.
MAX_ITER = 200 # maximum number of iterations to build coherence graphs
MAX_NODES = 500 # maximum number of nodes in coherence graphs
MAX_RECUR = 2 # maximum number of recursive calls in orienteering


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


def incCover(p, rest, vects, weights):
    """
    Incremental coverage of new path p over current map M.
    """
    other = rest + [p]
    return coverage(other, vects, weights) - coverage(rest, vects, weights)


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


def buildCoherenceGraph(pool, faces, times):
    """
    Form coherence graph G using photos from pool, where each node is
    an M-length coherent path and an edge exists if the nodes overlap
    by M-1 steps. Every node has coherence at least TAU. We do this
    using the general best-first search strategy employed in metro
    maps.  Returns list of nodes (chains) and adjacency matrix.
    """
    # priority queue of subchains
    pq = Queue.PriorityQueue()
    # nodes of the graph
    nodes = []

    # Fill queue with pairs with sufficiently high coherence
    for i in pool:
        for j in pool:
            if i != j:
                c = coherence([i,j], faces, times)
                if c > TAU:
                    pq.put((-c, [i,j])) #negative because we want max coherence first

    iter = 0
    while not pq.empty() and iter < MAX_ITER and len(nodes) < MAX_NODES:
        # Expand the chain with highest coherence using all possible extensions
        coh, chain = pq.get()
        # shuffle to add randomness
        shuffle(pool)
        for i in pool:
            if i not in chain:
                newChain = chain + [i]
                c = coherence(newChain, faces, times)
                if c > TAU:
                    # If we reach length m, make a new node
                    if len(newChain) >= M:
                        nodes.append(newChain)
                    else:
                        pq.put((-c, newChain))
        iter += 1

    # Add edges to the graph
    n = len(nodes)
    edges = np.eye(n)
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            u = nodes[i]
            v = nodes[j]
            # If overlap by m-1, add an edge
            hasEdge1 = hasEdge2 = True
            for k in range(1, M):
                if u[k] != v[k-1]:
                    hasEdge1 = False
                if v[k] != u[k-1]:
                    hasEdge2 = False
            if hasEdge1:
                edges[i, j] = 1
            if hasEdge2:
                edges[j, i] = 1

    return nodes, edges

    
def RG(s, t, B, rest, nodes, edges, bPaths, vects, weights, maxRecur):
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
            return p, incCover(p, rest, vects, weights)
        # Otherwise infeasible
        return [], 0.0
    p = []
    c = 0.0
    
    # Guess middle node and cost to reach it, and recurse
    # Note that the only candidate middle nodes are those within B from both s and t
#    guesses = set(np.nonzero(bPaths[s])[0]) & set(np.nonzero(bPaths[:,t])[0]) #set intersect
    for v in np.arange(len(nodes)):
        for b in np.arange(1, B+1):
            # If either of these are infeasible, try another b
            p1, c1 = RG(s, v, b, rest, nodes, edges, bPaths, vects, weights, maxRecur-1)
            if len(p1) == 0:
                continue
            p2, c2 = RG(v, t, B-b, rest + p1, nodes, edges, bPaths, vects, weights, maxRecur-1)
            if len(p2) == 0:
                continue

            newC = incCover(p1 + p2, rest, vects, weights)
            if newC > c:
                p = p1 + p2[1:] #start at 1 to omit the shared node
                c = newC
    return p, c


def getCoherentPath(map, nodes, edges, faces, times, places, weights):
    """
    Return a coherent path in G = (nodes, edges) that maximize
    coverage.  We accomplish this through submodular orienteering with
    recursion depth MAX_RECUR to find maximum coverage paths between
    every two nodes in G, then we greedily choose the best path to add
    to the map and repeat.
    """
    B = NUM_PHOTOS - M
    vects = np.hstack([faces, times, places])
    # Count paths <= B length in graph
    bPaths = np.empty(edges.shape)
    np.copyto(bPaths, edges)
    for i in np.arange(1, B):
        bPaths += bPaths.dot(edges)
            
    # Find best-coverage paths using orienteering
    maxPath = []
    maxCov = 0.0
    maxConn = connectivity(map, faces)
    items = np.arange(len(nodes))
    for s in items:
        for t in items:
            p, c = RG(s, t, B, [], nodes, edges, bPaths, vects, weights, MAX_RECUR)
            if (len(maxPath) == 0 and c > maxCov) or c > maxCov + EPSILON:
                print p
                maxPath = p
                maxCov = c
            elif len(maxPath) != 0 and abs(c - maxCov) < EPSILON:
                # If close to max coverage, only choose if greater connectivity
                newConn = connectivity(map + [p], faces)
                if newConn > maxConn:
                    print p
                    maxPath = p
                    maxCov = c
                    maxConn = newConn
    return maxPath


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
    for cl in whichClusters: #TODO low quality clusters => low coherence?
        pool = list(set(np.nonzero(faces[:,cl])[0])) #photos containing these faces

#        nodes, edges = buildCoherenceGraph(pool, faces, times)
#        print len(nodes), "nodes"
#        path = getCoherentPath(map, nodes, edges, faces, times, places, weights)
#        map.append(list(set(cbook.flatten(path))))

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
