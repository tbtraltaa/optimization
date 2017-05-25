import time
import numpy as np

from random_graph import random_graph

from bfs import bfs
from SAPMaxFlow import SAPMaxFlow
from FIFOPreflow import FIFOPreflow
from CSMaxFlow import CSMaxFlow

import matplotlib.pyplot as plt


def s_t_path_exists(pred, s, t):
    '''
    Check if there exists a s-t path.
    '''
    j = t
    while True:
        i = pred[j]
        if i == s:
            return True
        if i == 0:
            return False
        j = i

def M_analysis(k, n, m_min=None, m_max=None, U=10, l=None, s=1, t=None):
    l = int(n/3) + 2
    t = n
    if m_min is None:
        m_min = 2*n
    if m_max is None:
        m_max = 3*n
    M_list = np.linspace(2*n, np.floor(3*n), k)
    runtime = np.ndarray((3, k))
    Oruntime = np.ndarray((3, k))
    for i in np.arange(k):
        while True:
            while True:
                FStar = random_graph(n, M_list[i], l, U=U)
                if FStar is None:
                    continue
                order, pred = bfs(FStar, s, t)
                if s_t_path_exists(pred, s, t):
                    break

            start = time.time()
            SAPMaxFlow(FStar, s, t, False)
            runtime[0][i] = time.time() - start
            Oruntime[0][i] = n*n*M_list[i]
            
            start = time.time()
            CSMaxFlow(FStar, s, t, False)
            runtime[2][i] = time.time() - start
            Oruntime[2][i] = n*M_list[i]*np.log(U)

            start = time.time()
            x,y, z = FIFOPreflow(FStar, s, t, False)
            runtime[1][i] = time.time() - start
            Oruntime[1][i] = n**3
            if not z:
                continue
            else:
                print "================================"
                print "Iteration%d"%(i+1)
                print "SAP runtime: %.3f"%runtime[0][i]
                print "Scaling runtime: %.3f"%runtime[2][i]
                print "Preflow runtime: %.3f"%runtime[1][i]
                break

    plt.plot(M_list, runtime[0,:]*1000,'.-', label="SAP") 
    plt.plot(M_list, runtime[2,:]*1000, '.-', label="CS") 
    plt.plot(M_list, runtime[1,:]*1000, '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA runtime vs The number of arcs\n(n=%d, U=%d, m_min=%d, m_max=%d, k=%d)"\
    %(n,U, m_min, m_max, k))
    plt.xlabel("m - the number of arcs")
    plt.ylabel("Runtime in milliseconds")
    plt.show()

    plt.plot(M_list, Oruntime[0,:],'.-', label="SAP") 
    plt.plot(M_list, Oruntime[2,:], '.-', label="CS") 
    plt.plot(M_list, Oruntime[1,:], '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA Big-O vs The number of arcs\n(n=%d, U=%d, m_min=%d, m_max=%d, k=%d)"\
    %(n,U, m_min, m_max, k))
    plt.xlabel("m - the number of arcs")
    plt.ylabel("The number of steps")
    plt.show()

def MFA_AverageRuntime():
    '''
    Runtime analysis on Max Flow Algorithms
    '''
    #Number of arcs vs Runtime
    ###################################################
    k=20
    n = 20
    m_min = 20
    m_max = np.floor(3*n)
    M_analysis(k, n, m_min, m_max)
    ####################################################

if __name__ == "__main__":
    MFA_AverageRuntime()

