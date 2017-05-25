import numpy as np
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

def rand_series(n, k):
    '''
        Returns a sequence of k integers that sum up to n.
    '''
    if n > k:
        x_min = 1
        x_max = np.floor(n*1.0/k)
        if x_max > 10:
            x_min = 2
        x_list = np.random.randint(x_min, x_max+1, k) 
        delta = n - np.sum(x_list)
        if delta > k:
            x = np.floor(delta/k)
            x_list = x_list + x
            delta = delta - k*x
        if delta > 0:
            x_idx = np.random.choice(np.arange(int(k), dtype=int), size=delta, replace=False)
            x_list[x_idx] += 1
    else:
        x_list = np.zeros(k, dtype=int)
        x_idx = np.random.choice(np.arange(int(k), dtype=int), size=n, replace=False)
        x_list[x_idx] += 1
    return x_list.astype(int)

    
def get_layers(n_i):
    """
    Returns a dictionary of layers and nodes
    """
    layers = dict()
    idx = 1
    for i, val in enumerate(n_i):
        layers[i] = range(idx, int(idx+val))
        idx = idx + val
    return layers

def get_p(i, layers, marker):
    layer_marker = marker[layers[i]]
    if np.all(layer_marker==0):
        return None
    elif np.all(layer_marker==1):
        return None
    elif len(np.argwhere(layer_marker==0)) !=0:
        idx = np.random.choice(np.argwhere(layer_marker==0).reshape(-1,), size=1)[0]
        p = np.zeros(len(layer_marker))
        p[idx] = 1
        return p
    else:
        return None
    '''
    else:
        idx = np.argsort(layer_marker)
        if len(layer_marker) > 4:
            p = np.zeros(len(layer_marker))
            p[idx[-1]] = 0.5
            p[idx[-2]] = 0.5
        else:
            p = np.zeros(len(layer_marker))
            p[idx[-1]] = 1
        return p
   '''
def random_graph(n, m, l, U=10, p = np.array([0.1, 0.3, 0.2, 0.4]), U_min=1):
    """
    Generates a graph in Forward Star representaion 
    with n nodes, m arcs with l layers where
    n_i nodes, and m_i arcs correspond to i-th level.
    alpha - percentage of arcs going down 1-layer
    beta - percentage of arcs going down 2-layer
    gamma - percentage of arcs going down 3-layer
    mu - percentage of arcs going up and how many layers up is chosen randomly
    since those arcs are a few.
    """
    n_i = rand_series(n-2, l-2)
    
    min_m = n_i[0] + np.sum(n_i) + np.sum(n_i[:-1])+ 0
    tmp = n_i.copy()
    tmp[:-1] += n_i[:-1]
    m_i = np.concatenate(([n_i[0]], tmp, [0]))
    n_i = np.concatenate(([1], n_i, [1]))
    delta_m = m - min_m
    if delta_m < 0:
        print "m has to be at lease %d"%min_m
        exit()
    m_i[1:-2] = m_i[1:-2] + rand_series(delta_m, l-3)
    in_marker = np.zeros(n+1)
    out_marker = np.zeros(n+1)
    layers = get_layers(n_i)

    for i  in np.arange(l-2, -1, -1):
        # The second last layer
        delta = 0
        choices = [1, 2,3,-1]
        p_temp = p
        if i == l-2:
            FStar = np.array([layers[i][0], n])
            out_marker[layers[i][0]] += 1
            for idx, j in enumerate(layers[i][1:]):
                out_marker[j] += 1
                FStar = np.vstack((FStar, [j,n]))
            in_marker[n] = n_i[i]
            '''
            if m_i[i] > n_i[i]:
                delta = m_i[i] - n_i[i]
                p_temp = p[[0,3]]  
                p_temp[0] += p[1]+p[2] 
                print p_temp
                arc_types = np.random.choice([1, -1], size=m_i[i], replace=True, p=p_temp)
                offset = 1
            '''
        elif i == 0:
            for j in layers[1]:
                FStar = np.vstack((FStar, [1,j]))
                in_marker[j] += 1
                out_marker[0] += 1
            if m_i[i] > n_i[i]:
                delta = m_i[i] - n_i[i+1]
                choices = [2,3]
                p_temp = [p[0] +p[1] + p[3], p[2]]
        # All the remaining layers except the first layer
        else:
            LIST = [deepcopy(layers[i]), deepcopy(layers[i+1])]
            for idx in np.arange(n_i[i]):
                i_node = LIST[0][0]
                j_node = LIST[1][0]
                FStar = np.vstack((FStar, [i_node, j_node]))
                FStar = np.vstack((FStar, [j_node, i_node]))
                out_marker[i_node] += 1
                in_marker[i_node] += 1
                in_marker[j_node] += 1
                out_marker[j_node] += 1
                LIST[0].pop(0)
                LIST[0].append(i_node)
                LIST[1].pop(0)
                LIST[1].append(j_node)
            delta = m_i[i] - n_i[i]*2
            if  i <= l - 4:
                choices = [1,2,3,-1]
                p_temp = p  
            else:
                p_temp = p[[0,1,3]]  
                p_temp[1] += p[2] 
                choices = [1,2,-1]
        arc_types_cnt = 0
        while delta > 0:
            arc_types_cnt += 1
            if arc_types_cnt == 100:
                return None
            arc_types = np.random.choice(choices, size=delta, replace=True, p=p_temp)
            #arc_types = np.sort(arc_types)
            for idx, t in enumerate(arc_types):
                if i <= l-5:
                    offset = np.random.choice([1,2,3], size=1, replace=False)[0]
                else:
                    offset = np.random.choice([1,2], size=1, replace=False)[0]
                if t != -1:
                    prob = get_p(i, layers, out_marker)                
                    i_node= np.random.choice(layers[i], size=1, replace=False, p=prob)[0]
                else:
                    prob = get_p(i, layers, in_marker)                
                    j_node= np.random.choice(layers[i], size=1, replace=False, p=prob)[0]
                cnt = 0
                found = False
                unfound_cnt = 0
                while True :
                    if t != -1:
                        prob = get_p(i+t, layers, in_marker)                
                        j_node= np.random.choice(layers[i+t], size=1, replace=False, p=prob)[0]
                    else:
                        prob = get_p(i+offset, layers, out_marker)                
                        i_node= np.random.choice(layers[i+offset], size=1, replace=False, p=prob)[0]
                    if not np.any(np.all(FStar[:,0:2]==[i_node, j_node], axis=1)):
                        found = True
                        cnt = 0
                        break
                    cnt += 1
                    if cnt == 2:
                        break
                if not found:
                    unfound_cnt += 1
                    if unfound_cnt == 100:
                        return None
                    continue
                delta -= 1
                out_marker[i_node] += 1
                in_marker[j_node] += 1
                FStar = np.vstack((FStar, [i_node, j_node])) 

    # Adding a constant cost, 1
    FStar = np.hstack((FStar, np.ones((m,1), dtype=int)))
    # Adding random capacities
    FStar = np.hstack((FStar, np.random.randint(U_min, U+1, (m,1), dtype=int)))
    return FStar

if __name__ == "__main__":
    n = int(30)
    m = int(40)
    l = int(n/3)
    U = 100
    random_graph(n, m, l+2, U=10)
