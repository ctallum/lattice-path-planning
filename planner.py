"""
File for taking the graph and crating path and gcode
"""

import networkx as nx
from networkx import Graph
from typing import Dict, List, Tuple
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import deque

class TreeNode(object):
    def __init__(self, pos: Tuple[float, float] = None, parent = None, children: List = None, is_boundary = False):
        self.pos = pos
        self.parent = parent
        self.children = children if children is not None else []
        self.is_boundary = is_boundary
    
    def __hash__(self) -> int:
        return hash(self.pos[0]) + hash(self.pos[1])

    def __eq__(self, other):
         return (
             self.__class__ == other.__class__ and
             self.pos == other.pos and
             self.parent == other.parent
        )
 

class Planner:
    def __init__(self, params: Dict, layer_polygons: List[List[Polygon]], layer_graphs: List[List[nx.Graph]]) -> None:
        """
        Initialize planner
        """
        self.layer_graphs = layer_graphs
        self.params = params
        self.layer_polygons = layer_polygons
        self.n_layers = len(layer_graphs)
    
    def plan(self):
        """
        For all layers, generate spanning tree, then path, then generate gcode
        """
        for layer_idx in range(self.n_layers):
            for region_idx, graph in enumerate(self.layer_graphs[layer_idx]):
                polygon = self.layer_polygons[layer_idx][region_idx]
                tree = self.generate_spanning_tree(polygon, graph)

    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph):
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside
        """

        poly_points = (polygon.get_xy().tolist())
        

        for start_node, attributes in graph.nodes(data=True):
            for idx in range(len(poly_points) - 1):
                p1 = tuple(poly_points[idx])
                p2 = tuple(poly_points[idx + 1])
                p3 = tuple([attributes["x"],attributes["y"]])

                if are_collinear(p1,p2,p3):
                    break
        # insert break
        poly_points =  [[attributes["x"],attributes["y"]]] +  poly_points[idx+1:-1] + poly_points[0:idx+1] +  [[attributes["x"],attributes["y"]]]

        root = TreeNode(pos=poly_points[0], is_boundary=True)
        cur = root

        
        for poly_point_idx in range(len(poly_points) - 1):

            next_points = poly_points[poly_point_idx + 1]
            
            new_node = TreeNode(pos=next_points, is_boundary=True, parent=cur)
            
            cur.children.append(new_node)
            
            cur = new_node 

        # dfs through netwrokx graph

        visited = set()  # Set to keep track of visited nodes
        tree_edges = []  # List to store edges of the spanning tree


        def dfs(node, parent):
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor != parent:
                    if neighbor not in visited:
                        tree_edges.append((node, neighbor))
                        dfs(neighbor, node)
                    else:
                        # Ensure each node is visited at most twice
                        if (node, neighbor) not in tree_edges and (neighbor, node) not in tree_edges:
                            tree_edges.append((node, neighbor))

        dfs(start_node, None)

        unique_nodes = {}
        # get unique values
        for edge in tree_edges:
            node_1, node_2 = edge
            if node_1 not in unique_nodes.keys():
                unique_nodes[node_1] = TreeNode()
            if node_2 not in unique_nodes.keys():
                unique_nodes[node_2] = TreeNode()
        

        for edge in tree_edges[1:]:
            parent_idx, child_idx = edge
            parent = unique_nodes[parent_idx]
            child = unique_nodes[child_idx]

            if child.parent:
                child = TreeNode()

            parent_pos = [graph.nodes[parent_idx]['x'],graph.nodes[parent_idx]['y']]
            child_pos = [graph.nodes[child_idx]['x'],graph.nodes[child_idx]['y']]

            parent.pos = parent_pos
            child.pos = child_pos

            parent.children.append(child)
            child.parent = parent


        cur.children.append(unique_nodes[tree_edges[0][1]])    

        self.plot_tree(root)
        
    
    def plot_tree(self, node: TreeNode) -> None:
        for child in node.children:
            plt.plot([node.pos[0], child.pos[0]],[node.pos[1],child.pos[1]])
            self.plot_tree(child)
            

    def generate_path(self, params):
        pass

    def generate_gcode(self, params):
        pass

    

def are_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate slopes
    slope1 = (y2 - y1) * (x3 - x2)
    slope2 = (y3 - y2) * (x2 - x1)

    # print(slope1,slope2)

    # If slopes are equal, points are collinear
    return abs(slope1 - slope2) <.00001