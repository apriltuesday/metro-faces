# Code modded to work with current version, but using old metro maps coherence.

def incCover(p, M, xs, weights):
    """
    Incremental coverage of new path p over current map M, using xs as features
    weighted by weights
    """
    other = M + [p]
    return coverage(other, xs, weights) - coverage(M, xs, weights)


def CELF(paths, candidates, xs, weights):
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
        cov = incCover(p[0], paths, xs, weights)
        # If previously computed, we've found the best path
        if p[1]:
            paths.append(p[0])
            break
        else:
            candidates.put((-cov, (p[0], True)))


def buildCoherenceGraph(pool, xs, times, m=3, maxIter=10):
    """
    Form coherence graph G, where each node is an m-length coherent
    path and an edge exists if the nodes overlap by m-1 steps. Every
    node has coherence at least TAU. We do this using the general
    best-first search strategy employed in metro maps.
    Returns list of nodes (chains) and adjacency matrix.
    """
    items = pool #np.arange(xs.shape[0])
    # priority queue of subchains
    pq = Queue.PriorityQueue()
    # nodes of the graph
    nodes = []

    # Fill queue with singletons
    for i in items:
        pq.put((0, [i]))

    iter = 0
    while not pq.empty() and iter < maxIter:
        # Expand the chain with highest coherence using all possible extensions
        coh, chain = pq.get()
        # shuffle to add randomness
        shuffle(items)
        for i in items:
            if i not in chain:
                newChain = chain + [i]
                c = coherence(newChain, xs, times)
                if c > TAU:
                    # If we reach length m, make a new node
                    if len(newChain) >= m:
                        nodes.append(newChain)
                    else:
                        pq.put((-c, newChain)) #negative because we want max coherence first
        iter += 1

    # Add edges to the graph
    n = len(nodes)
    edges = np.eye(n) # connect nodes to selves...
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


def RG(s, t, B, paths, nodes, edges, bPaths, xs, weights, maxRecur=5):
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
            return p, incCover(p, paths, xs, weights)
        # Otherwise infeasible
        return [], 0.0
    P = []
    m = 0.0
    
    # Guess middle node and cost to reach it, and recurse
    # Note that the only candidate middle nodes are those within B from both s and t
    guesses = set(np.nonzero(bPaths[s])[0]) & set(np.nonzero(bPaths[:,t])[0]) #set intersect
    for v in guesses:
        for b in np.arange(1, B+1):
            # If either of these are infeasible, try another b
            p1, c1 = RG(s, v, b, paths, nodes, edges, bPaths, xs, weights, maxRecur-1)
            if len(p1) == 0:
                continue
            p2, c2 = RG(v, t, B-b, paths + [p1], nodes, edges, bPaths, xs, weights, maxRecur-1)
            if len(p2) == 0:
                continue

            newM = incCover(p1 + p2, paths, xs, weights)
            if newM > m:
                P = p1 + p2[1:] #start at 1 to omit the shared node
                m = newM
    return P, m


def getCoherentPath(pool, nodes, edges, paths, xs, weights, maxRecur=5):
    """
    Return one coherent paths in G = (nodes, edges) that
    maximizes coverage.  We accomplish this through submodular
    orienteering with recursion depth maxRecur to find maximum coverage paths
    between every two nodes in G, then we greedily choose the best
    path to add to the map and repeat.
    """
    if len(nodes) == 0:
        return []

    imgs = pool
    B = NUM_TIMES - len(nodes[0])
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

    # Use orienteering to get list of candidate paths
    for im1 in imgs:
        starts = beginsWith[im1]
        for im2 in imgs:
            if im1 == im2:
                continue
            ends = endsWith[im2]
            maxPath = []
            maxCov = 0.0

            # Find best-coverage path between each pair of images
            for s in starts:
                for t in ends:
                    p, c = RG(s, t, B, paths, nodes, edges, bPaths, xs, weights, maxRecur)
                    if c > maxCov:
                        maxPath = p
                        maxCov = c
                        
                # we keep 1 candidate per pair of images
            if len(maxPath) > 0:
                print maxPath
                candidates.put((-maxCov, (maxPath, False))) #false is for use by celf

    # Greedily choose best candidate
    CELF(paths, candidates, xs, weights)

    # Flatten paths and return map
    # (I wrote this all out to make it super explicit and less confusing)
    M = []
    for p in paths:
        newP = []
        for node in p:
            for img in node:
                if img not in newP:
                    newP.append(img)
        M.append(newP)
    return M
