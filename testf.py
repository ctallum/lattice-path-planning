from itertools import product, combinations
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from slicer import Slicer
from random import random
import pickle
import collections
from matplotlib.backend_bases import KeyEvent


def cpickle():
    f = open('full_graphs.pckl','rb')
    h = pickle.load(f)
    f.close()
    #print(h)
    return h

def uvec(vector):
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    v1_u = uvec(v1)
    v2_u = uvec(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def posgen(G):
    ret = {}
    for n in G:
        #print(n)
        #print(G.nodes)
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
def colgen(G):
    ret = []
    #print(len(G.edges),"aaaaa")
    for i,n in enumerate(G.edges):
        #print(n)
        #print(G.edges)
        if n[0] <= -1 or n[1] <= -1:
            ret.append('lightgreen')
        else:
            ret.append('darkblue')
    return ret

def angC(G: nx.Graph, a: tuple, b:tuple):
    #print(a,b, "CALLED ANGC")
    rr = tuple(set(a) & set(b))
    #print(rr, "TWO EDGES")
    rra1 = invr(a.index(rr[0]))
    rrb1 = invr(b.index(rr[0]))
    ooox = np.array([G.nodes[rr[0]]["x"], G.nodes[rr[0]]["y"]])
    p1 = np.array([G.nodes[a[rra1]]["x"], G.nodes[a[rra1]]["y"]])
    p2 = np.array([G.nodes[b[rrb1]]["x"], G.nodes[b[rrb1]]["y"]])
    #print(p1,ooox,p2)
    p1=(p1-ooox)
    p2=(ooox-p2)
    #print(p1,p2,"p1p2")
    p1=uvec(p1)
    p2=uvec(p2)
    #angle = np.arctan2(p1,p2)[0]
    angle = np.arctan2(np.cross(p1, p2), np.dot(p1, p2))
    #print(angle)
    return angle

def outb(a: list, b: tuple):
    #print(a, "BEFORE REM")
    #print(b, "tuple to be removed")
    if b == None: return a
    c=(b[1],b[0])
    if b in a:
        a.remove(b)
        return a.copy()
    elif c in a:
        a.remove(c)
        return a.copy()
    else: return a.copy()
    
def gover(a: tuple, b: int):
    if a[0] == b:
        return a[1]
    else: return a[0]

def invr(a: int):
    if a==1:
        return 0
    elif a==0:
        return 1

data1 = [(0,{"x": 0, "y": 0}),
         (1,{"x": 1, "y": 0}),
         (2,{"x": 2, "y": 0}),
         (3,{"x": 0, "y": 1}),
         (4,{"x": 1, "y": 1}),
         (5,{"x": 2, "y": 1}),
         (6,{"x": 0, "y": 2}),
         (7,{"x": 1, "y": 2}),
         (8,{"x": 2, "y": 2}),
         (9,{"x": 3, "y":3})
         ]

data2 = [(0,1),
         (1,2),
         (0,3),
         (1,4),
         (2,5),
         (3,4),
         (4,5),
         (3,6),
         (4,7),
         (5,8),
         (6,7),
         (7,8),
         (7,9),
         (8,9),
         (6,9),
         (1,5),
         (2,9)
         ]

def which_paths(x : int, seed : int): #TODO optional way to pick a path out of 2^(x) paths
    if seed >= 2**x: return None
    tf = [False, True]
    return list(product(tf, repeat=x))[seed]

def create_path(G: nx.Graph, tree: nx.Graph, width: float, seed):
    l_ed = []
    AA = nx.Graph()
    for u,v in G.edges:
        if not tree.has_edge(u,v):
            l_ed.append((u,v))
    ch_b = [True] * len(l_ed)
    ch_b = which_paths(len(l_ed), seed)
    for i in range(len(l_ed)):
        #u = l_ed[i][0]
        #v = l_ed[i][1]
        if ch_b[i] == True:
            unx = np.array([G.nodes[l_ed[i][0]]["x"]-G.nodes[l_ed[i][1]]["x"],G.nodes[l_ed[i][0]]["y"]-G.nodes[l_ed[i][1]]["y"]])
            unx_hat = unx / np.linalg.norm(unx)
            G.add_node(-(i+1), **({"x": (unx_hat*width)[0]+G.nodes[l_ed[i][1]]["x"], "y": (unx_hat*width)[1]+G.nodes[l_ed[i][1]]["y"]}) )
            G.add_edge(l_ed[i][0], -(i+1))
        else:
            unx = np.array([G.nodes[l_ed[i][1]]["x"]-G.nodes[l_ed[i][0]]["x"],G.nodes[l_ed[i][1]]["y"]-G.nodes[l_ed[i][0]]["y"]])
            unx_hat = unx / np.linalg.norm(unx)
            G.add_node(-(i+1), **({"x": (unx_hat*width)[0]+G.nodes[l_ed[i][0]]["x"], "y": (unx_hat*width)[1]+G.nodes[l_ed[i][0]]["y"]}) )
            G.add_edge(l_ed[i][1], -(i+1))
    G.remove_edges_from(l_ed)
    degrees = np.array([val for (node, val) in G.degree()])[0:len(G.nodes)-len(l_ed)]
    uniques, dcount = np.unique(degrees, return_counts=True)
    try:
        verxcount = dict(zip(uniques, dcount))[1]
    except KeyError:
        verxcount = 0
    traj = []
    sn = 0 #sn
    crnn = sn
    cuee = G.edges(sn)
    icm_e = None
    LEFTORRIGHT = True # true if left
    ixix = 0
    lastcycle = False
    lverxcount = 0
    allturnc = 0
    while (True and ixix < 35):
        if lastcycle: break
        if crnn == sn and ixix>4 and  (((lverxcount == verxcount and verxcount != 0) or (verxcount == 0 and LEFTORRIGHT==True)) and (verxcount+len(l_ed) == allturnc)): lastcycle = True
        ava_e = outb(list(cuee), icm_e)
        #print(ava_e, "avae")
        #print(icm_e, "icme")
        if icm_e == None:
            traj.append(crnn)
            crnn = gover(ava_e[0],crnn)
            cuee = G.edges(crnn)
            icm_e = ava_e[0]
            continue
        if len(ava_e) == 0: 
            LEFTORRIGHT = not LEFTORRIGHT
            if crnn >= 0: lverxcount+=1
            allturnc+=1
            traj.append(crnn)
            traj.append("TURN")
            #print("TURNING")
            icm_e = None
            #print(crnn, "crnn val rn")
            #print(icm_e, "icm val rn")
            #print(ava_e, "ava val rn")
            continue
              
        anglepicks = np.zeros(len(ava_e))
        for edge in range(len(ava_e)):
            #print(edge, "AAAAAAAA")
            anglepicks[edge] = angC(G, ava_e[edge], icm_e)
        #print(anglepicks, "ANGLE PICKS")
        #print(ava_e, "AVAE AFTER")
        indextogo = anglepicks.argmin()
        traj.append(crnn)
        crnn = gover(ava_e[indextogo], crnn)
        cuee = G.edges(crnn)
        icm_e = ava_e[indextogo]
        ixix+=1
    print(traj, "CALCULATED TRAVERSAL ORDER STARTING AT {A}".format(A=sn))
    #print("THESE ARE THE DEGREES: ", degrees)
    return traj
              

def create_gcode(): #TODO
    pass     
        
def n_stree(G) -> int:
    if not nx.is_connected(G):
        return -1
    n_nodes = G.number_of_nodes()
    l_mat = nx.laplacian_matrix(G).toarray()
    c_mat = l_mat[1:n_nodes, 1:n_nodes]
    det_val = np.linalg.det(c_mat)
    return int(np.round(abs(det_val)))

def recurse_cont_del(t_i, p_edges, spanning_trees, tec):
    if tec == 0:
        spanning_trees.extend(product(*t_i))
    
    for i in range(tec):
        if p_edges[tec][i] == []: continue
        t_i.append(p_edges[tec][i])
        p_edges[i] = [p_edges[i][j] + p_edges[tec][j] for j in range(i)]
        recurse_cont_del(t_i, p_edges, spanning_trees, tec-1)
        t_i.pop()
        [p_edges[i][j].pop() for j in range(i) for tt in range(len(p_edges[tec][j]))]
    
def spanning_trees(G):
    if not nx.is_connected(G):
        return -1
    n_nodes = G.number_of_nodes()
    a_edges = list(G.edges)
    p_edges = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
    nc_edges = len(a_edges)
    d_edges = dict()

    for edg in a_edges:
        i,j = sorted(edg)
        p_edges[j][i] = [nc_edges]
        d_edges[nc_edges] = edg
        nc_edges = nc_edges - 1
    array_to_fill = []
    recurse_cont_del([], p_edges, array_to_fill, n_nodes-1)
    return [nx.Graph(d_edges[i] for i in e) for e in array_to_fill]



# def ptp(tr : list, wd : float, G : nx.Graph):
#     wside = False
#     tr.pop() # last element same as first
#     lnx = len(tr)
#     rlist = []
#     for nr in range(lnx):
#         if tr[nr] == 'TURN':
#             pass
#         else:
#             x1 = G.nodes[tr[(nr-1) % lnx]]["x"]
#             y1 = G.nodes[tr[(nr-1) % lnx]]["y"]
#             x2 = G.nodes[tr[nr]]["x"] 
#             y2 = G.nodes[tr[nr]]["y"]
#             x3 = G.nodes[tr[(nr+1) % lnx]]["x"]
#             y3 = G.nodes[tr[(nr+1) % lnx]]["y"]
#             av = np.array([x1,y1])
#             bv = np.array([x2,y2])
#             cv = np.array([x3,y3])
#             cpx = (av+bv+cv)/3
            
        
G = nx.Graph()
G.add_nodes_from(data1)
G.add_edges_from(data2)


# FA = cpickle()[0]
# FA = nx.convert_node_labels_to_integers(FA,first_label=0)
# print("Num of Possible Unique Spanning Trees:", n_stree(FA))

#print("DBG: ", end='')
SDX = 0
all_span = spanning_trees(G)
#create_path(G, all_span[SDX], 0.2, 1)
#create_path(G, nx.minimum_spanning_tree(G), 0.5, which_paths)
closer = False
printedX = False
pix=0
def on_close(event):
    global closer
    closer = True
def next_t(event):
    global SDX
    global pix
    global printedX
    if event.button == 3:
        pix = 0
        SDX+=1
        printedX = False
def next_rt(event):
    global SDX
    global pix
    global printedX
    if type(event) == KeyEvent:
        pix += 1
        printedX = False
        return
    elif event.button == 3:
        pix += 1
        printedX = False
        return
   
fig = plt.figure(1, figsize = 1.0*np.array([12,6]))
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('key_press_event', next_rt)

while not closer:
    if not printedX:
        plt.clf()
        G.clear()
        G.add_nodes_from(data1)
        G.add_edges_from(data2)
        plt.subplot(121)
        pos = posgen(G)
        nx.draw(G, pos, with_labels=True, node_color='red', node_size=100, edge_color='#00AAFF')
        trajectory = create_path(G, all_span[SDX], 0.2, pix)
        pos = posgen(G)
        plt.subplot(122)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=100, edge_color=colgen(G))
        nx.draw(all_span[SDX], pos, with_labels=True, node_color='green', node_size=100, edge_color='r')
        printedX = True
        axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut = Button(axcut, 'Current relay index: {A}'.format(A = pix), color='pink', hovercolor='lightgreen')
        bcut.label.set_fontsize(7)
        bcut.on_clicked(next_rt)
        
        axcut2 = plt.axes([0.8, 0.0, 0.1, 0.075])
        bcut2 = Button(axcut2, 'Current tree index: {A}'.format(A = SDX), color='lightblue', hovercolor='lightgreen')
        bcut2.label.set_fontsize(7)
        bcut2.on_clicked(next_t)
    
    plt.pause(1/240) #refresh rate
    
#nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, edge_color=colgen(G))
#plt.show()