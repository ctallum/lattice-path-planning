from itertools import product
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def posgen(G):
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret

G = nx.Graph()

data1 = [(0,{"x": 0, "y": 0}),
         (1,{"x": 1, "y": 0}),
         (2,{"x": 2, "y": 0}),
         (3,{"x": 0, "y": 1}),
         (4,{"x": 1, "y": 1}),
         (5,{"x": 2, "y": 1}),
         (6,{"x": 0, "y": 2}),
         (7,{"x": 1, "y": 2}),
         (8,{"x": 2, "y": 2})]

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
         (7,8)
         ]


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
        # print(i,j)
        p_edges[j][i] = [nc_edges]
        d_edges[nc_edges] = edg
        nc_edges = nc_edges - 1
    array_to_fill = []
    recurse_cont_del([], p_edges, array_to_fill, n_nodes-1)
    return [nx.Graph(d_edges[i] for i in e) for e in array_to_fill]


G.add_nodes_from(data1)
G.add_edges_from(data2)

print("Num of Possible Unique Spanning Trees:", n_stree(G))
print(len(G.edges))
print(len(G.nodes))


all_span = spanning_trees(G)
print(all_span)

pos = posgen(G)


ccxc = 0

def onclickA(event):
    global ccxc
    plt.clf()
    nx.draw(G, pos, with_labels=True, node_color='blue', node_size=200, edge_color='b')
    nx.draw(all_span[ccxc], pos, with_labels=True, node_color='green', node_size=200, edge_color='r')
    ccxc += 1
    plt.show() 

fig,ax = plt.subplots()
bax = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnbtn = Button(bax, "")
bnbtn.on_clicked(onclickA)

#nx.draw(G, pos, with_labels=True, node_color='blue', node_size=200, edge_color='b')
#nx.draw(all_span[0], pos, with_labels=True, node_color='green', node_size=200, edge_color='r')
plt.show()
