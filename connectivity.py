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
