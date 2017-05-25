import numpy as np

# Forward star representation [T, H, C, U]
# [0, 0, 0] is a dummy node.
FStar = np.array([[0, 0, 0],
                    [1, 2, 3],
                    [1, 3, 2],
                    [2, 3, 4],
                    [2, 6, 5],
                    [3, 4, 1],
                    [3, 7,-4],
                    [4, 2,-1],
                    [4, 5, 5],
                    [5, 6,-2],
                    [6, 7,-4],
                    [7, 5, 6],])

d = np.array([0, 0, 0, 2, 6, 8, 6, 4])

def adjacency_list(FStar):
    '''
    Outarc adjacency list
    '''
    A = dict()
    # Excludes the dummy node 0.
    N = np.sort(np.unique(FStar[:,0:2].reshape(-1,)))[1:]
    for i in N:
        A[i] = FStar[:,1][np.argwhere(FStar[:,0]==i)].reshape(-1,).tolist()
    return A

def FIFOlc(FStar, s, d=None):
    '''
    FIFO label correcting algorithm
    '''
    A = adjacency_list(FStar)
    n = len(A)
    NODELIST = list([s])
    if d is None:
        d = np.ndarray((n+1,))
        d[:] = np.inf
        d[s] = 0
    pred = np.zeros(n+1, dtype=int)
    pred[s] = 0
    while len(NODELIST) != 0:
        i = NODELIST.pop(0) #Removes the first element
        for j in A[i]:
            c = FStar[np.all(FStar[:,0:2]==[i,j], axis=1),2]
            if d[j] > d[i] + c:
                d[j] = d[i] + c
                pred[j] = i
                if j not in NODELIST:
                    NODELIST.append(j)
    return pred, d

if __name__ == "__main__":
    pred, d = FIFOlc(FStar, 1)
    print "Pred vector, pred:"
    print pred[1:]
    print "Distance vector, d:"
    print d[1:]
    

