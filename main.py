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


NUM_LINES = 4
NUM_CLUSTERS = 80 #???

def coverage(map, xs):
    """
    Computes coverage of the set of chains map.  See
    metromaps for details.
    """
    subset = list(set(cbook.flatten(map)))
    # Compute coverages, vector-style
#    vects = np.hstack([faces, times, places]) #XXX regions
#    weights = np.hstack([faceWeights, timeWeights, placeWeights])
    weights = np.ones(xs.shape[1])
    total = (1 - np.prod(1 - xs[subset], axis=0)).dot(weights)
    return total


def greedy(map, candidates, xs):
    """
    Greedily choose the max-coverage candidate and add to map.
    """
    maxCoverage = 0.0
    maxP = 0
    for p in candidates:
        c = coverage(map + [p], xs)
        if c > maxCoverage:
            maxCoverage = c
            maxP = p
    map.append(maxP)


if __name__ == '__main__':
#    args = sys.argv

    # Load data
    mat = io.loadmat('../data/April_full_new.mat')
    images = mat['images'][:,0]
    faces = mat['facesBinary'] 
#    years = mat['timestamps'].reshape(n)
#    longitudes = mat['longitudes'].reshape(n)
#    latitudes = mat['latitudes'].reshape(n)
#    faces = mat['faces'] + mat['facesBinary']

    # Get a master list of names from contacts.xml
    names = []
    for line in open('../data/faces/contacts.xml'):
        try:
            rest = line[line.index('name=')+6:]
            name = rest[:rest.index('"')]
            names.append(name)
        except ValueError:
            continue
    names = np.array(names)

    # Omit people who don't appear
    names = names[~np.all(faces == 0, axis=0)]
    faces = faces[:, ~np.all(faces == 0, axis=0)]

    n, m = faces.shape
    photos = np.arange(n)
    ppl = np.arange(m)
    print 'done loading'

    # Form adjacency matrix of the social graph
    A = np.array([np.sum(np.product(faces[:,[i, j]], axis=1)) for i in ppl for j in ppl]).reshape((m,m))

    # Graph that sucker
    colors = cm.get_cmap(name='Spectral')
    G = nx.Graph(A)
    nodes = [A[i,i] for i in ppl]
    edges = []
    for (u,v) in G.edges():
        edges.append(A[u,v])

    #nx.draw(G, node_color=nodes, edge_color=edges, cmap=colors, edge_cmap=colors)
    layout = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=layout, node_size=[x*2 for x in nodes])
    nx.draw_networkx_edges(G, pos=layout, alpha=0.5, node_size=0, width=edges, edge_color='b')

    # Cluster faces
    c = cluster.bicluster.SpectralCoclustering(n_clusters=NUM_CLUSTERS)
    c.fit(A)
    clusters = [] #list of lists, each of which lists face indices in that cluster
    for i in range(NUM_CLUSTERS):
        clusters.append(c.get_indices(i)[0])

    # Choose clusters of faces to use for lines
    whichClusters = []
    for i in range(NUM_LINES):
        greedy(whichClusters, clusters, faces.T)
    #whichClusters is a list of lists, each of which lists face indices in that cluster
    for i in range(NUM_LINES):
        for j in whichClusters[i]:
            print j, names[j]
        print '------'

    # For each face cluster, choose representative photos (TODO)

    plt.show()
