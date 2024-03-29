import networkx as nx
import matplotlib.pyplot as plt

def posgen(G):
    ret = {}
    for n in G:
        print(n)
        ret[n] = [G.nodes[n]['x'],G.nodes[n]['y']]
    return ret

G = nx.Graph()

data1 = [(1,{"x": 0, "y": 0}),
         (2,{"x": 1, "y": 0}),
         (3,{"x": 2, "y": 0}),
         (4,{"x": 0, "y": 1}),
         (5,{"x": 1, "y": 1}),
         (6,{"x": 2, "y": 1}),
         (7,{"x": 0, "y": 2}),
         (8,{"x": 1, "y": 2}),
         (9,{"x": 2, "y": 2})
         ]
data2 = [(1,2),
         (2,3),
         (1,4),
         (2,5),
         (3,6),
         (4,5),
         (5,6),
         (4,7),
         (5,8),
         (6,9),
         (7,8),
         (8,9)
         ]

print("Node Count: ",len(data1))
print("Edge Count: ",len(data2))

G.add_nodes_from(data1)
G.add_edges_from(data2)
SP = nx.minimum_spanning_tree(G)

nx.draw(SP, pos=posgen(G), with_labels=True)
plt.show()

