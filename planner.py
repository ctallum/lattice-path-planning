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
from shapely import LineString, LinearRing
from shapely import buffer
import shapely as shp


class TreeNode(object):
    def __init__(self, pos: Tuple[float, float] = None, node = None, parent = None, children: List = None, is_boundary = False):
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
                tree =self.generate_spanning_tree(polygon, graph)
                point_path = self.generate_path(tree)

                line = LineString(point_path)
                offset_path = np.array(line.buffer(.5*self.params["line_width"]).exterior.coords)
                print(offset_path)


                plt.plot(*offset_path.T,"-b")
                plt.axis('equal')
                

    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph) -> TreeNode:
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside
        """

        # get buffer polygon that is 1x width smaller than original polygon
        poly_points = polygon.get_xy()

        buffer_poly = shp.Polygon(poly_points).buffer(-self.params["line_width"])
        buffer_poly_points = np.array(buffer_poly.exterior.coords.xy).T
        
        polygon = Polygon(buffer_poly_points)
        poly_points = polygon.get_xy()

        # plt.plot(*poly_points.T)

        
        # get list of connected graphs
        connected_components = list(nx.connected_components(graph))

        

        # for each subgraph, find an end piece
        end_nodes = []
        
        for sub_graph in connected_components:
            for node in sub_graph:
                if graph.degree(node) == 1:
                    end_nodes.append(node)
                    break

        # for each end node, add an edge that connects them to the outside
        new_end_nodes = []
        insert_idxs = [(0,0) for _ in end_nodes]

        for node_idx, node in enumerate(end_nodes):
            prev_node = next(graph.neighbors(node))
            
            # get line that extends outward
            line_1 = [(graph.nodes[prev_node]["x"],graph.nodes[prev_node]["y"]), 
                      (graph.nodes[node]["x"],graph.nodes[node]["y"])]
            best_dist = math.inf
            best_point = None
            for idx in range(len(poly_points) - 1):    
                line_2 = [tuple(poly_points[idx]), tuple(poly_points[idx+1])]

                # find best intersection point
                point = intersect_in_range(line_1, line_2)
                if point:
                    x,y = point
                    dist = math.sqrt((line_1[1][0] - x)**2 + (line_1[1][1] - y)**2)
                    if dist < best_dist:
                        best_point = point
                        best_dist = dist
                        insert_idxs[node_idx] = (idx, idx+1)

            new_end_nodes.append(best_point)


        graph_n_nodes = max(graph.nodes)


        # add outer edge to graph with new points
        new_points = []
        for idx in range(len(poly_points)-1):
            x1 = poly_points[idx][0]
            y1 = poly_points[idx][1]

            new_points.append((idx + graph_n_nodes + 1,{'x':x1, 'y':y1}))
        
        for new_end_node_idx, new_end_node in enumerate(new_end_nodes):
            x1, y1 = new_end_node
            new_points.append((idx + graph_n_nodes + new_end_node_idx + 2, {"x":x1, "y":y1}))

        graph.add_nodes_from(new_points)

        

        new_edges = []
        for vert_idx in range(len(poly_points) - 2):
            offset = graph_n_nodes +1
            if (vert_idx , vert_idx+1) not in insert_idxs:
                new_edges.append((vert_idx+offset, vert_idx+offset+1))
            else:
                sub_graph_idx = insert_idxs.index((vert_idx, vert_idx+1))
                new_edges.append((vert_idx + offset, graph_n_nodes + idx + sub_graph_idx + 2))
                new_edges.append((vert_idx + offset + 1, graph_n_nodes + idx + sub_graph_idx + 2))
                new_edges.append((end_nodes[sub_graph_idx],graph_n_nodes + idx + sub_graph_idx + 2))

        new_edges.append((max(graph.nodes) - len(new_end_nodes)- len(poly_points)+2, max(graph.nodes) - len(new_end_nodes)))

        graph.add_edges_from(new_edges)

        print(len(new_end_nodes))
   
        # nx.draw(graph, pos=posgen(graph), node_size = 1, with_labels=False)

            
        # find a starting point and remove close points        

        def is_valid_enter_pt(start_idx):
            x_1 = graph.nodes[start_idx]["x"]
            y_1 = graph.nodes[start_idx]["y"]
            for point in new_end_nodes:
                x_2,y_2 = point
                if math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2) < self.params["line_width"]:
                    return False
            return True

        potential_start_idx = max(graph.nodes) - len(new_end_nodes)
        while not is_valid_enter_pt(potential_start_idx):
            potential_start_idx -= 1

        # delete all close points to enter point
        del_node = potential_start_idx - 1
        cur_node = potential_start_idx
        while True:
            x_1 = graph.nodes[potential_start_idx]["x"]
            y_1 = graph.nodes[potential_start_idx]["y"]
            x_2 = graph.nodes[del_node]["x"]
            y_2 = graph.nodes[del_node]["y"]
            

            # Delete edges
            graph.remove_edges_from([(del_node, cur_node)])
            dist = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
            if dist > 2*self.params["line_width"]:

                vec_a = np.array([x_1,y_1]) - np.array([x_2, y_2])
                vec_a = vec_a/np.linalg.norm(vec_a) * (dist - 2*self.params["line_width"])
                new_point = (x_2 + vec_a[0], y_2 + vec_a[1])
                
                graph.add_nodes_from([(max(graph.nodes) + 1, {"x": new_point[0], "y":new_point[1]})])
                graph.add_edges_from([(del_node, max(graph.nodes))])
                

                break
            del_node -= 1
            cur_node -= 1

        graph.remove_nodes_from(list(nx.isolates(graph)))
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


        dfs(potential_start_idx, None)

        # # create a dictionary to hold the nodes of the lattice
        unique_nodes = {}

        # # get unique nodes and insert into dictionary
        for edge in tree_edges:
            node_1, node_2 = edge
            if node_1 not in unique_nodes.keys():
                unique_nodes[node_1] = TreeNode()
            if node_2 not in unique_nodes.keys():
                unique_nodes[node_2] = TreeNode()
        
        
        # using tree_edge list, establish connectivity between nodes
        for edge in tree_edges:
            parent_idx, child_idx = edge
            parent = unique_nodes[parent_idx]
            child = unique_nodes[child_idx]

            parent_pos = (graph.nodes[parent_idx]['x'],graph.nodes[parent_idx]['y'])
            offset_table = {"triangle":2.31, "square":2, "hexagon":2}
            # if the node has already been used, create new one at same spot, but now leaf
            if child.parent:
                child = TreeNode()
                child_pos = (graph.nodes[child_idx]['x'],graph.nodes[child_idx]['y'])
                x_1, y_1 = parent_pos
                x_2, y_2 = child_pos
                vec_a = np.array([x_2, y_2]) - np.array([x_1,y_1])
                length = np.linalg.norm(vec_a)
                vec_a = vec_a/length * (length - offset_table[self.params["infill"]]*self.params["line_width"])
                child_pos = (x_1 + vec_a[0], y_1 + vec_a[1])

            else:
                child_pos = (graph.nodes[child_idx]['x'],graph.nodes[child_idx]['y'])


            
            parent.pos = parent_pos
            child.pos = child_pos

            parent.children.append(child)
            child.parent = parent     

        root = unique_nodes[potential_start_idx]
        
        self.plot_tree(root)   

        return root

    
    def plot_tree(self, node: TreeNode) -> None:
        for child in node.children:
            plt.plot([node.pos[0], child.pos[0]],[node.pos[1],child.pos[1]],"-k")
            self.plot_tree(child)
            

    def generate_path(self, tree: TreeNode) -> Tuple[List[Tuple], List[TreeNode]]:
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

    

def line_equation(point1, point2):
    """
    Calculate the slope (m) and y-intercept (c) of the line passing through two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Check if the line is vertical (undefined slope)
    if x2 - x1 == 0:
        slope = None
        intercept = x1  # x-intercept
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

    return slope, intercept

def intersection_point(line1, line2):
    """
    Calculate the intersection point of two lines.
    """
    m1, c1 = line_equation(line1[0], line1[1])
    m2, c2 = line_equation(line2[0], line2[1])
    
    # Check if lines are parallel
    if m1 == m2:
        # Check if both lines are vertical and coincident
        if m1 is None and m2 is None and c1 == c2:
            return "Coincident"
        else:
            return None  # Lines are parallel, no intersection
    
    # Check if line1 is vertical
    if m1 is None:
        x = c1
        y = m2 * x + c2
        return x, y
    
    # Check if line2 is vertical
    if m2 is None:
        x = c2
        y = m1 * x + c1
        return x, y
    
    # Calculate intersection point
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    
    return x, y

def intersect_in_range(line1, line2):
    x,y = intersection_point(line1, line2)
    
    # Check if projected point lies on line2
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Check if projected point lies within the bounding box of line2
    if min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
        return x,y
    else:
        return None

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

def posgen(G):
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
