import numpy as np

FStar = np.array([[0, 0],
                    [1,2],
                    [1,3],
                    [1,5],
                    [2,4],
                    [2,5],
                    [3,5],
                    [3,6],
                    [4,5],
                    [4,8],
                    [5,6],
                    [6,7],
                    [6,9],
                    [7,4],
                    [7,5],
                    [7,8],
                    [9,7],
                    [9,8]]);
point = np.array([0, 1, 4, 6, 8, 10, 11, 13, 16, 16, 18])

def adjacency_list(FStar):
    '''
    Outarc adjacency list
    '''
    A = dict()
    N = np.sort(np.unique(FStar.reshape(-1,)))
    for i in N:
        A[i] = FStar[:,1][np.argwhere(FStar[:,0]==i)].reshape(-1,).tolist()
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
        n = len(A)
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
    index = np.arange(1, n+1, 1).reshape(-1,1)
    '''
    print "Order\n[Index, order]"
    print np.hstack((index, np.array(order[1:]).reshape(-1,1))).astype(int)
    print "Predecessor\n[Index, pred]"
    print np.hstack((index, np.array(pred[1:]).reshape(-1,1))).astype(int)
    print "Used lexicographical ordering to choose a sibling node to traverse first."
    '''
    return order, pred

if __name__ == "__main__":
    order, pred = bfs(FStar, 1)
    print "Order"
    print order
    print "Pred"
    print pred
