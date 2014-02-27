# April Shen
# Metro maps X Photobios - Baselines
#!/usr/bin/python

import numpy as np
from random import choice, sample


def yearsBaseline(years, k):
    """
    Bin by years into k bins, choose one at random from each.
    """
    n = years.shape[0]
    vertices = np.arange(n)

    # Some years are nan; for these, use mean year
    m = np.mean(np.ma.masked_where(np.isnan(years), years))
    years = [m if np.isnan(y) else y for y in years]

    # Split years in k bins, choose one from each
    sortedYears = sorted(vertices, key=lambda i: years[i])
    num = int(n / k) + 1
    bins = [sortedYears[x:x+num] for x in range(0, n, num)]
    yearsSubset = [choice(y) for y in bins]
    return yearsSubset


def kmedoids(features, k):
    """
    Compute cluster assignments using k-medoids.
    """
    n = features.shape[0]
    vertices = np.arange(n)

    # compute pairwise distances using feature vectors
    distances = np.zeros((n,n))
    for i in vertices:
        for j in np.arange(i+1, n):
            distances[i,j] = distances[j,i] = np.linalg.norm(features[i] - features[j])
    
    iters = 0    
    centers = sample(vertices, k)
    changed = True
    
    while changed and iters < 50:
        changed = False
        # calculate cluster assignments based on centers
        clusters = [[] for i in range(k)]
        for x in vertices:
            c = np.argmin(distances[x][centers])
            clusters[c].append(x)

        # calculate medoid with min cost
        for c in range(k):
            bestCenter = centers[c]
            bestCost = float('inf')
            for x in clusters[c]:
                cost = np.sum(distances[x][clusters[c]])
                if cost < bestCost:
                    bestCenter = x
                    bestCost = cost
            if bestCenter != centers[c]:
                centers[c] = bestCenter
                changed = True
        iters += 1

    return centers
