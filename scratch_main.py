
# Parameters of the algorithm
M = 3 # number of images in each node of coherence graph
TAU = 0.2 # This is the minimum coherence constraint.
EPSILON = 0.001 # This governs the connectivity/coverage tradeoff.
MAX_ITER = 200 # maximum number of iterations to build coherence graphs
MAX_NODES = 500 # maximum number of nodes in coherence graphs
MAX_RECUR = 2 # maximum number of recursive calls in orienteering

def incCover(p, rest, vects, weights):
    """
    Incremental coverage of new path p over current map M.
    """
    other = rest + [p]
    return coverage(other, vects, weights) - coverage(rest, vects, weights)



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
