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
import math

# import sys
# sys.setrecursionlimit(5000)

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
                path = self.generate_path(tree)
                # print(path)
                for point in path:
                    plt.plot(*point,'o')
                    plt.pause(.01)



    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph) -> TreeNode:
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside
        """

        poly_points = (polygon.get_xy().tolist())


        # find a starting node on edge that intersects with infill
        for start_node, attributes in graph.nodes(data=True):
            for idx in range(len(poly_points) - 1):
                p1 = tuple(poly_points[idx])
                p2 = tuple(poly_points[idx + 1])
                p3 = tuple([attributes["x"],attributes["y"]])

                if are_collinear(p1,p2,p3):
                    break
        
        # create new polygon that starts and stops at intersection
        poly_points =  [[attributes["x"],attributes["y"]]] +  poly_points[idx+1:-1] + poly_points[0:idx+1] +  [[attributes["x"],attributes["y"]]]

        # create tree root
        root = TreeNode(pos=poly_points[0], is_boundary=True)
        cur = root

        # loop around the border and add to tree
        for poly_point_idx in range(len(poly_points) - 1):
            next_points = poly_points[poly_point_idx + 1]
            new_node = TreeNode(pos=next_points, is_boundary=True, parent=cur)
            cur.children.append(new_node)
            cur = new_node 

        # work through lattice and create a spanning tree vis DPS

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

        # create a dictionary to hold the nodes of the lattice
        unique_nodes = {}

        # get unique nodes and insert into dictionary
        for edge in tree_edges:
            node_1, node_2 = edge
            if node_1 not in unique_nodes.keys():
                unique_nodes[node_1] = TreeNode()
            if node_2 not in unique_nodes.keys():
                unique_nodes[node_2] = TreeNode()
        
        # connect the first visited node to the end of the loop tree
        lat_root = unique_nodes[tree_edges[0][1]]
        lat_root.parent = cur
        cur.children.append(lat_root) 

        # using tree_edge list, establish connectivity between nodes
        for edge in tree_edges[1:]:
            parent_idx, child_idx = edge
            parent = unique_nodes[parent_idx]
            child = unique_nodes[child_idx]


            # if the node has already been used, create new one at same spot, but now leaf
            if child.parent:
                child = TreeNode()

            parent_pos = [graph.nodes[parent_idx]['x'],graph.nodes[parent_idx]['y']]
            child_pos = [graph.nodes[child_idx]['x'],graph.nodes[child_idx]['y']]

            parent.pos = parent_pos
            child.pos = child_pos

            parent.children.append(child)
            child.parent = parent     

        self.plot_tree(root)   

        return root
           
        
        
    
    def plot_tree(self, node: TreeNode) -> None:
        for child in node.children:
            plt.plot([node.pos[0], child.pos[0]],[node.pos[1],child.pos[1]])
            self.plot_tree(child)
            

    def generate_path(self, tree: TreeNode) -> List:
        points = []

        def dfs(node) -> None:
            points.append(node.pos)

            # calc relative angles to next point
            rel_angles = []
            if len(node.children) > 1:
                for child in node.children:
                    v_0 = np.array(node.pos).reshape((2, 1))
                    v_1 = np.array(node.parent.pos).reshape((2, 1))
                    v_2 = np.array(child.pos).reshape((2, 1))
                    rel_angles.append(angle(v_0,v_1,v_2)) 

                # reorder children
                node.children = [x for _, x in sorted(zip(rel_angles, node.children), key=lambda pair: pair[0])]

            for child in node.children:
                dfs(child)

            if node.parent:
                points.append(node.parent.pos)


        dfs(tree)
        
        return points


    def generate_gcode(self, params):
        pass

    

def are_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate slopes
    slope1 = (y2 - y1) * (x3 - x2)
    slope2 = (y3 - y2) * (x2 - x1)

    # If slopes are equal, points are collinear
    return abs(slope1 - slope2) <.00001

def angle(vertex0, vertex_1, vertex_2, angle_type='unsigned'):
    """
    Compute the angle between two edges  vertex0-- vertex_1 and  vertex0--
    vertex_2 having an endpoint in common. The angle is computed by starting
    from the edge  vertex0-- vertex_1, and then ``walking'' in a
    counterclockwise manner until the edge  vertex0-- vertex_2 is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex_1 - vertex0
    vec2 = vertex_2 - vertex0

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex_1 or vertex_2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    c_angle = np.inner(vec1flat, vec2flat)  # cos(theta) between two edges
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle