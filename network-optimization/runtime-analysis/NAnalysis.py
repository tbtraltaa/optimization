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

def N_analysis(k, n_min, n_max, m_scale=2, U=10, l=None, s=1, t=None):
    N_list = np.linspace(n_min, n_max, k).astype(int)
    runtime = np.ndarray((3, k))
    Oruntime = np.ndarray((3, k))
    for i in np.arange(k):
        n = N_list[i]
        m = np.floor(m_scale*n)
        l = int(n/3) + 2
        t = n
        while True:
            while True:
                FStar = random_graph(N_list[i], m, l, U=U)
                if FStar is None:
                    continue
                order, pred = bfs(FStar, s, t)
                if s_t_path_exists(pred, s, t):
                    break
            np.savetxt("FStar.txt", FStar, fmt="%d")
            start = time.time()
            SAPMaxFlow(FStar, s, t, False)
            runtime[0][i] = time.time() - start
            Oruntime[0][i] = (N_list[i]**2)*m
            
            start = time.time()
            CSMaxFlow(FStar, s, t, False)
            runtime[2][i] = time.time() - start
            Oruntime[2][i] = N_list[i]*m*np.log(U)

            start = time.time()
            x,y,z = FIFOPreflow(FStar, s, t, False)
            runtime[1][i] = time.time() - start
            Oruntime[1][i] = N_list[i]**3
            if not z:
                continue
            else:
                print "================================"
                print "Iteration%d"%(i+1)
                print "SAP runtime: %.3f"%runtime[0][i]
                print "Scaling runtime: %.3f"%runtime[2][i]
                print "Preflow runtime: %.3f"%runtime[1][i]
                break

    plt.plot(N_list, runtime[0,:]*1000,'.-', label="SAP") 
    plt.plot(N_list, runtime[2,:]*1000, '.-', label="CS") 
    plt.plot(N_list, runtime[1,:]*1000, '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA runtime vs The number of nodes\n(n_min=%d, n_min=%d, U=%d, k=%d)"\
    %(n_min, n_max,U,k))
    plt.xlabel("n - the number of nodes")
    plt.ylabel("Runtime in milliseconds")
    plt.show()

    plt.plot(N_list, Oruntime[0,:],'.-', label="SAP") 
    plt.plot(N_list, Oruntime[2,:], '.-', label="CS") 
    plt.plot(N_list, Oruntime[1,:], '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA Big-O time vs The number of nodes\n(n_min=%d, n_max=%d, U=%d, k=%d)"\
    %(n_min, n_max, U, k))
    plt.xlabel("n - the number of nodes")
    plt.ylabel("The number of steps")
    plt.show()

def MFA_AverageRuntime():
    '''
    Runtime analysis on Max Flow Algorithms
    '''
    #Number of nodes vs Runtime
    ###################################################
    k = 20
    n_min = 10
    n_max = 40
    N_analysis(k, n_min, n_max, m_scale=3)
    ####################################################

if __name__ == "__main__":
    MFA_AverageRuntime()

