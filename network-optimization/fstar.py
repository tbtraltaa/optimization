import numpy as np

def main():
    fstar = np.array([  [0, 0, 0, 0, 0],
                        [1, 1, 2, 5, np.inf],
                        [2, 1, 3, 3, np.inf],
                        [3, 2, 4, -2, 10],
                        [4, 3, 2, -1, 20],
                        [5, 3, 5, 10, np.inf],
                        [6, 5, 4, 2, np.inf]]).reshape(-1,5)
    point = np.array([[1, 1], 
                     [2, 3], 
                     [3, 4], 
                     [4, 6], 
                     [5, 6], 
                     [6, 7]]).reshape(-1,2)
    print "Forward star representation of the graph in Figure 2.28:"
    print "The first column for the both table is the indexing."
    print """Arc data(The first row is a dummy row since the indexing in "point vector" starts from 1.):"""
    print fstar
    print "Point vector:"
    print point

    bstar = np.array([  [0, 0, 0, 0, 0],
                        [1, 1, 2, 5, np.inf],
                        [2, 3, 2, -1, 20],
                        [3, 1, 3, 3, np.inf],
                        [4, 2, 4, -2, 10],
                        [5, 5, 4, 2, np.inf],
                        [6, 3, 5, 10, np.inf]]).reshape(-1,5)
    rpoint = np.array([[1, 1], 
                     [2, 1], 
                     [3, 3], 
                     [4, 4], 
                     [5, 6], 
                     [6, 7]]).reshape(-1,2)
    print "=============================================================================================="
    print "Backward star representation of the graph in Figure 2.28:"
    print "The first column for the both table is the indexing."
    print """Arc data(The first row is a dummy row since the indexing in "rpoint vector" starts from 1.):"""
    print bstar
    print "rpoint vector:"
    print rpoint

if __name__ == "__main__":
    main()
