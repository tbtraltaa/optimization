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

def U_analysis(k, n, m, Umin, Umax, l=None, s=1, t=None):
    l = int(n/3) + 2
    t = n
    U_list = np.linspace(Umin, Umax, k)
    runtime = np.ndarray((3, k))
    Oruntime = np.ndarray((3, k))
    for i in np.arange(k):
        while True:
            while True:
                FStar = random_graph(n, m, l, U=U_list[i])
                if FStar is None:
                    continue
                order, pred = bfs(FStar, s, t)
                if s_t_path_exists(pred, s, t):
                    break
            start = time.time()
            SAPMaxFlow(FStar, s, t, False)
            runtime[0][i] = time.time() - start
            Oruntime[0][i] = n*n*m
            
            start = time.time()
            CSMaxFlow(FStar, s, t, False)
            runtime[2][i] = time.time() - start
            Oruntime[2][i] = n*m*np.log(U_list[i])

            start = time.time()
            x, y, z = FIFOPreflow(FStar, s, t, False)
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

    plt.plot(U_list, runtime[0,:]*1000,'.-', label="SAP") 
    plt.plot(U_list, runtime[2,:]*1000, '.-', label="CS") 
    plt.plot(U_list, runtime[1,:]*1000, '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA runtime vs Capacity(n=%d, m=%d, k=%d)"%(n,m,k))
    plt.xlabel("U-Maximum Capacity")
    plt.ylabel("Runtime in milliseconds")
    plt.show()

    plt.plot(U_list, Oruntime[0,:],'.-', label="SAP") 
    plt.plot(U_list, Oruntime[2,:], '.-', label="CS") 
    plt.plot(U_list, Oruntime[1,:], '.-', label="Preflow") 
    plt.legend(loc="upper left")
    plt.title("MFA Big-O time vs Capacity(n=%d, m=%d, k=%d)"%(n,m,k))
    plt.xlabel("U-Maximum Capacity")
    plt.ylabel("Runtime in number of steps")
    plt.show()

def MFA_AverageRuntime():
    '''
    Runtime analysis on Max Flow Algorithms for
    varying U.
    '''
    # Max Capacity vs Runtime
    ###################################################
    n = int(20)
    m = int(60)
    Umin = 1
    Umax = 1500 
    k = 20
    U_analysis(k, n, m, Umin, Umax)
    ####################################################


if __name__ == "__main__":
    MFA_AverageRuntime()

