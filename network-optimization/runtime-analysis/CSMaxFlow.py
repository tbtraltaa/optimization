import numpy as np


# Forward star representation [T, H, C, U]
# [0, 0, 0] is a dummy node.
FStar = np.array([  [1, 2, 1, 3],
                    [1, 3, 1, 3],
                    [1, 4, 1, 2],
                    [2, 5, 1, 4],
                    [3, 4, 1, 1],
                    [3, 6, 1, 2],
                    [4, 2, 1, 1],
                    [4, 6, 1, 2],
                    [5, 4, 1, 1],
                    [5, 6, 1, 1]]).reshape(-1, 4)

def adjacency_list(FStar):
    '''
    Outarc adjacency list
    '''
    A = dict()
    # Excludes the dummy node 0.
    N = np.sort(np.unique(FStar[:,0:2].reshape(-1,)))
    G = FStar[np.where(FStar[:,3]>0)]
    for i in N:
        A[i] = list(np.sort(G[:,1][np.argwhere(G[:,0]==i)].reshape(-1,)).tolist())
    return A


def bfs(FStar, s, n=None):
    '''
    Breadth First Search using Forward Star representation.
    FStar - Forward Star representation
    s - source node
    '''
    A = adjacency_list(FStar)
    # Number of nodes
    if n is None:
        n = np.unique(FStar[:,[0,1]].reshape(-1,)).shape[0]
    # Tails of arcs
    T = FStar[:,0]
    H = FStar[:,1]
    # Heads of arcs
    nodesToTraverse = list([s])
    mark = np.zeros(n+1)
    next_= 1
    mark[s] = 1
    pred = np.zeros(n+1)
    pred[s] = 0
    order = np.zeros(n+1)
    order[s] = next_
    while len(nodesToTraverse) != 0:
        i = nodesToTraverse[0]
        # Unmarked j nodes of (i,j) arcs 
        jj = []
        if i in A.keys():
            if A[i] != []:
                jj = [j for j in A[i] if mark[j]==0]
                # Lexicographical ordering
                jj.sort()
            if len(jj) !=0:
                for j in jj:
                    mark[j] = 1
                    pred[j] = i
                    next_ += 1
                    order[j] = next_
                    nodesToTraverse.append(j)
            else:
                nodesToTraverse.pop(0)
        else:
            nodesToTraverse.pop(0)
    '''
    index = np.arange(1, n+1, 1).reshape(-1,1)
    print "Order\n[Index, order]"
    print np.hstack((index, np.array(order[1:]).reshape(-1,1))).astype(int)
    print "Predecessor\n[Index, pred]"
    print np.hstack((index, np.array(pred[1:]).reshape(-1,1))).astype(int)
    print "Used lexicographical ordering to choose a sibling node to traverse first."
    '''
    return order, pred

def is_admissible(A, i, NA):
    if len(A[i])!= 0:
        for j in A[i]:
            if j not in NA:
                return True
    return False

def s_t_path_exists(pred, s, t):
    '''
    Check if there exists a s-t path exists
    '''
    j = t
    while True:
        i = pred[j]
        if i == s:
            return True
        if i == 0:
            return False
        j = i

def augment(RFStar, RFStar_delta, A, pred, s, t):
    j = t
    delta = np.inf
    P = [t]
    while j != s and j != 0:
        i = pred[j]
        P.insert(0, i)
        u = RFStar_delta[np.all(RFStar_delta[:,0:2]==[i,j], axis=1),3][0]
        if u < delta:
            delta = u
        j = i
    for idx, i  in enumerate(P[:-1]):
        u_ij = RFStar_delta[np.all(RFStar_delta[:,0:2]==[i,P[idx+1]], axis=1),3]
        if u_ij > delta:
            RFStar[np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
            RFStar_delta[np.all(RFStar_delta[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
        if u_ij == delta:
            RFStar_delta[np.all(RFStar_delta[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
            RFStar[np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
            A[i].remove(P[idx+1])
            A[i].sort()
        '''
        else:
            np.delete(RFStar, np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1), 0)
        '''
        # If there exist the reverse arc
        if np.any(np.all(RFStar_delta[:,0:2]==[P[idx+1], i], axis=1)):
            u_ji = RFStar_delta[np.all(RFStar_delta[:,0:2]==[P[idx+1],i], axis=1),3]
            RFStar_delta[np.all(RFStar_delta[:,0:2]==[P[idx+1],i], axis=1),3] = u_ji + delta
            RFStar[np.all(RFStar[:,0:2]==[P[idx+1],i], axis=1),3] = u_ji + delta
        else:
            RFStar_delta = np.vstack((RFStar_delta, [P[idx+1], i, 1, delta]))
            A[P[idx+1]].append(i)
            A[P[idx+1]].sort()
            if np.any(np.all(RFStar[:,0:2]==[P[idx+1], i], axis=1)):
                u_ji = RFStar[np.all(RFStar_delta[:,0:2]==[P[idx+1],i], axis=1),3]
                RFStar[np.all(RFStar[:,0:2]==[P[idx+1],i], axis=1),3] = u_ji + delta
            else:
                RFStar = np.vstack((RFStar, [P[idx+1], i, 1, delta]))
    return RFStar, RFStar_delta, A

    
def CSMaxFlow(FStar, s, t, output=True):
    """
    Capacity scaling algorithm
    --------------------------
    """
    RFStar = FStar.copy()
    # Excludes the dummy node 0.
    N = np.sort(np.unique(RFStar[:,0:2].reshape(-1,)))[1:]
    # Number of nodes    n = len(N)
    x = np.zeros(RFStar.shape[0], dtype=int)
    U = np.amax(RFStar[:,3])
    n = len(np.unique(RFStar[:,0:2].reshape(-1,)))
    delta = int(np.floor(2.0**np.log(U)))
    while delta >= 1:
        RFStar_delta = RFStar[RFStar[:,3] >= delta]
        A = adjacency_list(RFStar_delta)
        pred = np.zeros(n+1, dtype=int)
        order, pred = bfs(RFStar_delta, s, n) 
        while s_t_path_exists(pred, s, t):
            RFStar, RFStar_delta, A = augment(RFStar, RFStar_delta, A, pred, s, t)
            order, pred = bfs(RFStar_delta, s, n) 
        delta = delta/2
    max_x = 0
    x = np.ndarray(shape=(FStar.shape[0],3), dtype=int)
    x[:,[0,1]] = FStar[:, [0,1]]
    for idx, row in enumerate(FStar):
        r_ij = RFStar[np.all(RFStar[:,0:2]==[row[0], row[1]], axis=1),3]
        if row[3] >= r_ij:
            x[idx,2] = row[3] - r_ij
        else:
            x[idx,2] = 0
        if row[0] == s:
            max_x += x[idx,2]
    if output:
        #Lexicographical ordering by T, H
        #The last key is the primary key
        idx = np.lexsort((RFStar[:,1].reshape(-1,), RFStar[:,0].reshape(-1,)))
        RFStar = RFStar[idx, :].astype(int)
        print "Capacity Scaling MaxFlow algorithm"
        print "=================================="
        print "Original network-[T H C U]:"
        print FStar
        print "Residual network-[T H C U]:"
        print RFStar
        print "The max flow, x-vector - [T H x]:"
        print x
        print "The max flow value:"
        print max_x
    return x, max_x 
if __name__ == "__main__":
    s=1
    t=6
    CSMaxFlow(FStar, 1, 6)
