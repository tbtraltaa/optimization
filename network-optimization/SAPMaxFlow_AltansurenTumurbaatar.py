import numpy as np


# Forward star representation [T, H, C, U]
# [0, 0, 0] is a dummy node.
FStar = np.array([[0, 0, 0, 0],
                    [1, 2, 1, 3],
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
    N = np.sort(np.unique(FStar[:,0:2].reshape(-1,)))[1:]
    for i in N:
        A[i] = np.sort(FStar[:,1][np.argwhere(FStar[:,0]==i)].reshape(-1,)).tolist()
    return A

def in_adjacency_list(FStar):
    '''
    Inarc adjacency list along with flow. The initial flow is zero.
    '''
    AI = dict()
    # Excludes the dummy node 0.
    N = np.sort(np.unique(FStar[:,0:2].reshape(-1,)))[1:]
    for j in N:
        nodes = np.sort(FStar[:,0][np.argwhere(FStar[:,1]==j)].reshape(-1,)).tolist()
        #[Tail node, x_ij]
        AI[j] = [i for i in nodes]
    return AI

def reverse_bfs(FStar, s):
    '''
    Breadth First Search using Reverse Star representation.
    FStar - Forward Star representation
    s - source node
    '''
    A = in_adjacency_list(FStar)
    # Number of nodes
    n = len(A)
    # Number of arcs
    m = FStar.shape[0] - 1
    # Tails of arcs
    T = FStar[:,0]
    H = FStar[:,1]
    # Heads of arcs
    nodesToTraverse = list([s])
    mark = np.zeros(n+1)
    next_= 1
    mark[s] = 1
    pred = np.zeros(n+1)
    pred[:] = np.inf
    pred[s] = 0
    order = np.zeros(n+1)
    order[:] = np.inf
    order[s] = next_
    while len(nodesToTraverse) != 0:
        j = nodesToTraverse[0]
        # Unmarked i nodes of (i,j) arcs 
        ii = [i for i in A[j] if mark[i]==0]
        # Lexicographical ordering
        ii.sort()
        if len(ii) !=0:
            for i in ii:
                mark[i] = 1
                pred[i] = j
                next_ += 1
                order[i] = next_
                nodesToTraverse.append(i)
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

def depth(pred):
    """
    Given a pred vector, returns depths of nodes in BFS tree as distance labels
    """
    n = len(pred) - 1
    d = np.ndarray((n+1,))
    root = np.argwhere(pred==0)[0][0]
    d[0] = np.inf
    for i in np.arange(n):
        #level
        l = 0
        node = i+1
        if pred[node] == np.inf:
            d[node] = np.inf
        else:
            while node != root:
                p = int(pred[node])
                node = p
                l += 1
            d[i+1] = l
    return d

def advance(A, pred, d, i):
    for j in A[i]:
        if d[i] == d[j] + 1:
            pred[j] = i
            i = j
            break
    return pred, i
    
def is_admissible(A, d, i):
    for j in A[i]:
        if d[i] == d[j] + 1:
            return True
    return False

def augment(RFStar, A, pred, s, t):
    j = t
    delta = np.inf
    P = [t]
    while j != s:
        i = pred[j]
        P.insert(0, i)
        u = RFStar[np.all(RFStar[:,0:2]==[i,j], axis=1),3][0]
        if u < delta:
            delta = u
        j = i

    for idx, i  in enumerate(P[:-1]):
        u_ij = RFStar[np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1),3]
        if u_ij > delta:
            RFStar[np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
        if u_ij == delta:
            RFStar[np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1),3] = u_ij-delta
            A[i].remove(P[idx+1])
            A[i].sort()
        '''
        else:
            np.delete(RFStar, np.all(RFStar[:,0:2]==[i,P[idx+1]], axis=1), 0)
        '''
        if np.any(np.all(RFStar[:,0:2]==[P[idx+1], i], axis=1)):
            u_ji = RFStar[np.all(RFStar[:,0:2]==[P[idx+1],i], axis=1),3]
            RFStar[np.all(RFStar[:,0:2]==[P[idx+1],i], axis=1),3] = u_ji + delta
        else:
            RFStar = np.vstack((RFStar, [P[idx+1], i, 1, delta]))
            A[P[idx+1]].append(i)
            A[P[idx+1]].sort()
    return RFStar, A

def retreat(RFStar, A, pred, d, i, s):
    min_d = np.inf
    if len(A[i]) != 0:
        for j in A[i]:
            r_ij = RFStar[np.all(RFStar[:,0:2]==[i, j], axis=1),3]
            if r_ij > 0 and (d[j] + 1) < min_d:
                min_d = d[j] + 1
        d[i] = min_d
        if i != s:
            i = pred[i]
    else:
        d[s] = np.inf
    return d, i
    
def SAP(FStar, s, t):
    """
    Shortest augmenting path algorithm
    ----------------------------------
    """
    RFStar = FStar.copy()
    order, bfs_pred = reverse_bfs(FStar[:,[0,1]], t)
    d = depth(bfs_pred)
    # Excludes the dummy node 0.
    N = np.sort(np.unique(FStar[:,0:2].reshape(-1,)))[1:]
    # Number of nodes
    n = len(N)
    x = np.zeros(FStar.shape[0], dtype=int)
    i = s
    pred = np.zeros(n+1, dtype=int)
    A = adjacency_list(RFStar)
    while d[s] < n:
        if is_admissible(A, d, i):
            pred, i = advance(A, pred, d, i)
            if i == t:
                RFStar, A = augment(RFStar, A, pred, s, t)
                i = s
        else:
            d, i = retreat(RFStar, A, pred, d, i, s)
    #Lexicographical ordering by T, H
    #The last key is the primary key
    idx = np.lexsort((RFStar[:,1].reshape(-1,), RFStar[:,0].reshape(-1,)))
    RFStar = RFStar[idx, :]
    print "Original network-[T H C U]:"
    print FStar
    print "Residual network-[T H C U]:"
    print RFStar
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
    print "The max flow, x-vector - [T H x]:"
    print x
    print "The max flow value:"
    print max_x
if __name__ == "__main__":
    s=1
    t=6
    SAP(FStar, s, t)
