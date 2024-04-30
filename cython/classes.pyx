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
from gcodepy.gcode import Gcode
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import trimesh 
from matplotlib.patches import Polygon
from tqdm import tqdm
import lattpy as lp
from lattpy import Lattice
import networkx as nx
from shapely.geometry import LineString
import shapely as shp
from collections import defaultdict
import pickle
import os


sys.setrecursionlimit(10000)

class TreeNode(object):
    def __init__(self, pos: Tuple[float, float] = None, node = None, parent = None, children: List = None, is_boundary = False):
        self.pos = pos
        self.node = node
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

        pbar = tqdm(total=self.n_layers,desc = "Planning Layer Path")
        layer_paths = []
        layer_full_graphs = []
        for layer_idx in range(self.n_layers):
            # layer_idx = 37
            region_paths = []

            
            
            for region_idx, graph in enumerate(self.layer_graphs[layer_idx]):
                polygon = self.layer_polygons[layer_idx][region_idx]

                
                


                tree, complete_graph =self.generate_spanning_tree(polygon, graph)
                # nx.draw(complete_graph, pos=posgen(graph), node_size = 1, with_labels=False)
                # self.plot_tree(tree[0])
                plt.axis("equal")

                # print(tree)
                
                
                # print(tree)
                if tree:
                    for root in tree:
                        point_path = self.generate_path(root)

                        # plt.plot(*np.array(point_path).T,"-b")
                        # plt.axis('equal')
                        
                        line = LineString(point_path)
                        offset_path = np.array(line.buffer(.5*self.params["line_width"]).exterior.coords)
                        # print(offset_path)
                        region_paths.append(offset_path)

                # for offset_path in region_paths:
                #     plt.plot(*offset_path.T,"-b")
                # plt.axis('equal')
            layer_paths.append(region_paths)
            
            layer_full_graphs.append(complete_graph)
            # return [], []
            pbar.update(1)

        
            
        
        return layer_paths, layer_full_graphs

    def generate_gcode(self, layer_paths):
        pbar = tqdm(total=self.n_layers,desc = "Generating gcode")
        extrusion = 0
        layer_num = 0 

        def generate_gcode_for_loop(layer_loops, extrusion, layer_num, feed_rate=100, retract_distance=.08, lift_distance=2):
            # global extrusion
            # global layer_num
            gcode = []
            gcode.append(f";LAYER:{layer_num}")
            for points in layer_loops:
                # Set feed rates
                gcode.append(f"G1 F2100 E{extrusion + .08}")
                gcode.append(f"G1 F{feed_rate}")  # Set X, Y, and Z feed rate
                
                
                # Generate G-code for loop
                for idx, point in enumerate(points):
                    x, y, z = point

                    # go to start point and drop to level
                    if idx == 0:
                        gcode.append(f"G0 X{x} Y{y}")
                        gcode.append(f"Go Z{z}")

                    else:
                        # calc distance moved
                        ratio = self.params["layer_height"]*self.params["line_width"] / ((1.75/2)**2*3.14)
                        dist = math.sqrt((x - old_x)**2 + (y - old_y)**2)*ratio
                        extrusion += dist

                        gcode.append(f"G1 X{x} Y{y} E{extrusion}")

                    old_x = x
                    old_y = y
                
                # retract filament a slight amount and move up Z 
                gcode.append(f"G1 F2100 E{extrusion - retract_distance}")
                gcode.append(f"G0 F{feed_rate} Z{z+lift_distance*self.params['layer_height']}")


            layer_num = layer_num + 1
                    
            return gcode, extrusion, layer_num          

        def generate_gcode_for_loops(loops, extrusion, layer_num, feed_rate=100):
            # global extrusion
            gcode = []
            
            # Initialize
            gcode.append("M82") # set absolute extrusion mode
            gcode.append("G21")  # Set units to millimeters
            gcode.append("G90")  # Set to absolute positioning
            gcode.append("G92 E0")  # Zero the extruder
            gcode.append("M82") # set absolute extrusion mode

            # set temps
            gcode.append("M104 S200")
            gcode.append("M140 S60")
            gcode.append("M190 S60")
            gcode.append("M109 S200")
            
            # Home
            gcode.append("G28")  # Home all axes
            gcode.append("G92 E0.0")
            gcode.append(f"G1 F2100 E{extrusion - .08}")
            
            # Set feed rates
            gcode.append(f"G1 F{feed_rate}")  # Set X, Y, and Z feed rate
            
            # Generate G-code for each loop
            gcode.append(f";LAYER_COUNT:{self.n_layers - 1}")
            for loop in loops:
                new_code, extrusion, layer_num = generate_gcode_for_loop(loop, extrusion, layer_num)
                gcode.extend(new_code)
                pbar.update(1)
                
            
            # End of program
            gcode.append("M140 S0")
            gcode.append("M107")
            gcode.append("M104 S0")
            gcode.append("M140 S0")
            gcode.append("M107")
            gcode.append("M84")
            gcode.append("M82")
            gcode.append("M30")  # End of program
            
            return "\n".join(gcode)
        
        loops = []

        for layer_idx, layer in enumerate(layer_paths):
            layer_loops = []
            layer_height = layer_idx * .2
            for array in layer:
                new_column = np.full((array.shape[0], 1), layer_height)

            # Horizontally stack the original array and the new column
                new_arr = np.hstack((array, new_column))
                layer_loops.append(new_arr)
            
            loops.append(layer_loops)
            



        # Generate G-code
        gcode_content = generate_gcode_for_loops(loops, extrusion, layer_num)

        # Save G-code to a file
        with open("output.gcode", "w") as file:
            file.write(gcode_content)



    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph) -> TreeNode:
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside
        """

        # get buffer polygon that is 1x width smaller than original polygon
        poly_points = polygon.get_xy()
        # plt.plot(*poly_points.T)
        # nx.draw(graph, pos=posgen(graph), node_size = 1, with_labels=False)

        

        buffer_poly = shp.Polygon(poly_points).buffer(-self.params["line_width"])

        poly_exteriors = []
        if type(buffer_poly) == shp.MultiPolygon:
            for sub_poly_idx in range(len(buffer_poly.geoms)):
                poly_exteriors.append(Polygon(np.array(buffer_poly.geoms[sub_poly_idx].exterior.coords.xy).T))
        else:
            poly_exteriors.append(Polygon(np.array(buffer_poly.exterior.coords.xy).T))

        # buffer_poly_points = np.array(buffer_poly.exterior.coords.xy).T
        
        # polygon = Polygon(buffer_poly_points)
        # poly_points = polygon.get_xy()

        poly_points_array = [polygon.get_xy() for polygon in poly_exteriors]

        for poly_points in poly_points_array:
            if not np.any(poly_points):
                return None, None
            
        
        
        # get list of connected graphs
        connected_components = list(nx.connected_components(graph))
        # print(len(connected_components))

        # for each subgraph, find an end piece
        end_nodes = []
        



        # print(end_nodes)

        # for each end node, add an edge that connects them to the outside
        new_end_nodes = []
        insert_idxs = [(0,0) for _ in connected_components]

        def find_connection(region_idx, node):
            # print("finding connection for node ", node)
            prev_node = next(graph.neighbors(node))
            
            # get line that extends outward
            line_1 = [(graph.nodes[prev_node]["x"],graph.nodes[prev_node]["y"]), 
                      (graph.nodes[node]["x"],graph.nodes[node]["y"])]
            best_dist = math.inf
            best_point = (None,None)
            idx_total = 0
            for poly_points in poly_points_array:
                # print("new range")
                for idx in range(len(poly_points) - 1):    
                    line_2 = [tuple(poly_points[idx]), tuple(poly_points[idx+1])]
                    # print(line_2)
                    # print(line_1)

                    # find best intersection point
                    point = intersect_in_range(line_1, line_2)
                    # print(point)
                    # print(line_2)
                    
                    # print(point)
                    if point:
                        x,y = point
                        dist = math.sqrt((line_1[1][0] - x)**2 + (line_1[1][1] - y)**2)
                        if dist < best_dist:
                            best_point = point
                            best_dist = dist
                            if idx_total + idx+1 == len(poly_points) - 1:
                                insert_idxs[region_idx] = (idx+idx_total, idx_total)
                            else:
                                insert_idxs[region_idx] = (idx+idx_total, idx_total + idx+1)
                idx_total += idx
            if abs(best_dist - 2*self.params["line_width"]) > .01:
                return (None, None)
            # print(best_dist)
            return best_point
            
        for region_idx, sub_graph in enumerate(connected_components):
            for node in sub_graph:
                if graph.degree(node) == 1:
                    # print("trying to find connection for node ", node)
                    connection_pt = find_connection(region_idx, node)
                    # print(connection_pt)
                    if not connection_pt[0] is None:
                        # print(connection_pt)
                        end_nodes.append(node)
                        new_end_nodes.append(connection_pt)
                        break
    
        # for node_idx, node in enumerate(end_nodes):
        #     prev_node = next(graph.neighbors(node))
            
        #     # get line that extends outward
        #     line_1 = [(graph.nodes[prev_node]["x"],graph.nodes[prev_node]["y"]), 
        #               (graph.nodes[node]["x"],graph.nodes[node]["y"])]
        #     best_dist = math.inf
        #     best_point = (None,None)
        #     idx_total = 0
        #     for poly_points in poly_points_array:
        #         for idx in range(len(poly_points) - 1):    
        #             line_2 = [tuple(poly_points[idx]), tuple(poly_points[idx+1])]
        #             # print(line_2)

        #             # find best intersection point
        #             point = intersect_in_range(line_1, line_2)
                    
        #             # print(point)
        #             if point:
        #                 x,y = point
        #                 dist = math.sqrt((line_1[1][0] - x)**2 + (line_1[1][1] - y)**2)
        #                 if dist < best_dist:
        #                     best_point = point
        #                     best_dist = dist
        #                     insert_idxs[node_idx] = (idx+idx_total, idx_total + idx+1)
        #         idx_total += idx

            # new_end_nodes.append(best_point)
        # print(new_end_nodes)

        if connected_components:
            graph_n_nodes = max(graph.nodes)
        else:
            graph_n_nodes = 0

        # print(insert_idxs)

        # add outer edge to graph with new points
        
        # offset = max(graph.nodes) + 1


        
        idx = 0
        new_points = []
        for poly_points in poly_points_array:
            

            # print(poly_points)
            
            for p_idx in range(len(poly_points)-1):
                x1 = poly_points[p_idx][0]
                y1 = poly_points[p_idx][1]

                new_points.append((idx + graph_n_nodes + 1,{'x':x1, 'y':y1}))
                # print(idx + graph_n_nodes + 1)

                idx += 1


            
        for new_end_node_idx, new_end_node in enumerate(new_end_nodes):
            x1, y1 = new_end_node
            new_points.append((idx + graph_n_nodes + new_end_node_idx + 1, {"x":x1, "y":y1}))
            
            # print(idx + graph_n_nodes + new_end_node_idx + 1)

        # print(new_points)
        graph.add_nodes_from(new_points)
        
        outside_pts = []

            # print(idx + graph_n_nodes + new_end_node_idx + 2)
        # print(new_points)
        # print(insert_idxs)
        # print(len(poly_points))
        # print(end_nodes)
        # print(new_end_nodes)

        # nx.draw(graph, pos=posgen(graph), node_size = 1, with_labels=True)
        # plt.axis("equal")

        

        new_edges = []

        # return None, None
        


        vert_idx = 0
        sub_offset = 0
        for poly_points in poly_points_array:
            regional_outside_pts = []
            for edge_idx in range(len(poly_points) - 2):   
                # print((vert_idx , vert_idx+1))  
                regional_outside_pts.append(vert_idx+graph_n_nodes+1+sub_offset)
                regional_outside_pts.append(vert_idx+graph_n_nodes+2+sub_offset)           
                if (vert_idx , vert_idx+1) not in insert_idxs and (vert_idx + 1, vert_idx) not in insert_idxs:

                    # print((vert_idx+graph_n_nodes+1, vert_idx+graph_n_nodes+2))
                    new_edges.append((vert_idx+graph_n_nodes+1+sub_offset, vert_idx+graph_n_nodes+2+sub_offset))
                    # print((vert_idx+offset, vert_idx+offset+1))
                else:
                    # continue
                    # print("hi")
                    sub_graph_idx = insert_idxs.index((vert_idx, vert_idx+1))
                    new_edges.append((vert_idx+graph_n_nodes+1+sub_offset, graph_n_nodes + idx + sub_graph_idx + 1))
                    new_edges.append((vert_idx+graph_n_nodes+2+sub_offset, graph_n_nodes + idx + sub_graph_idx + 1))
                    new_edges.append((end_nodes[sub_graph_idx],graph_n_nodes + idx + sub_graph_idx + 1))
                    regional_outside_pts.append(graph_n_nodes + idx + sub_graph_idx + 1)
                vert_idx +=1
            last_edge = ( vert_idx, vert_idx + 2 - len(poly_points))
            if last_edge not in insert_idxs and (last_edge[1],last_edge[0]) not in insert_idxs:
                new_edges.append((vert_idx+graph_n_nodes+1+sub_offset, vert_idx+graph_n_nodes+3+sub_offset - len(poly_points)))
            else:
                sub_graph_idx = insert_idxs.index(last_edge)
                new_edges.append((vert_idx+graph_n_nodes+1+sub_offset, graph_n_nodes + idx + sub_graph_idx + 1))
                new_edges.append((graph_n_nodes + idx + sub_graph_idx + 1, vert_idx+graph_n_nodes+3+sub_offset - len(poly_points)))
                new_edges.append((graph_n_nodes + idx + sub_graph_idx + 1, end_nodes[sub_graph_idx]))

                # print(vert_idx+graph_n_nodes+1+sub_offset)
                # print( vert_idx+graph_n_nodes+3+sub_offset - len(poly_points))
                # print(end_nodes[sub_graph_idx])
                # print(graph_n_nodes + idx + sub_graph_idx + 1)
            # print((vert_idx+graph_n_nodes+1+sub_offset, vert_idx+graph_n_nodes+3+sub_offset - len(poly_points)))
            sub_offset += 1
            outside_pts.append(regional_outside_pts)


            # new_edges.append((max(graph.nodes) - len(new_end_nodes)- len(poly_points)+2, max(graph.nodes) - len(new_end_nodes)))

        graph.add_edges_from(new_edges)
        # print(new_edges)
   
#         nx.draw(graph, pos=posgen(graph), node_size = 1, with_labels=False)
#         plt.axis("equal")
# # #           
        
        
        # find a starting point and remove close points

       
        used_pts = []
        # print(outside_pts)        

        # return None, None
    
        
        def get_valid_start_pt(outside_pts):
            for idx in outside_pts:
                # print(idx)
                # print(graph.degree(idx))
                if graph.degree(idx) == 3:
                    return idx
            # print(graph.degree(idx))
            return idx
        
        potential_start_idxs = []

        potential_start_idx = graph_n_nodes
        for region_idx, poly_points in enumerate(poly_points_array):
            # print(poly_points)
            # potential_start_idx += len(poly_points) -1
            # while not is_valid_enter_pt(potential_start_idx):
            #     potential_start_idx -= 1
            potential_start_idx = get_valid_start_pt(outside_pts[region_idx])
            
            # delete all close points to enter point
            # del_node = potential_start_idx - 1
            # print(list(graph.neighbors(potential_start_idx)))
            del_node = next(graph.neighbors(potential_start_idx))
            
            while del_node not in outside_pts[region_idx]:
                del_node = next(graph.neighbors(potential_start_idx))
            
            cur_node = potential_start_idx
            used_pts.append(cur_node)
            
            while True:
                x_1 = graph.nodes[potential_start_idx]["x"]
                y_1 = graph.nodes[potential_start_idx]["y"]
                x_2 = graph.nodes[del_node]["x"]
                y_2 = graph.nodes[del_node]["y"]
                

                # Delete edges
                graph.remove_edges_from([(del_node, cur_node)])
                # print((del_node, cur_node))

                # print((del_node, cur_node))
                dist = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                if dist > 2*self.params["line_width"]:
                    
                    vec_a = np.array([x_1,y_1]) - np.array([x_2, y_2])
                    vec_a = vec_a/np.linalg.norm(vec_a) * (dist - 2*self.params["line_width"])
                    new_point = (x_2 + vec_a[0], y_2 + vec_a[1])
                    
                    graph.add_nodes_from([(max(graph.nodes) + 1, {"x": new_point[0], "y":new_point[1]})])
                    graph.add_edges_from([(del_node, max(graph.nodes))])

                    # print(x_1,y_1)
                    # print("happening", max(graph.nodes))
                    

                    break
                cur_node = del_node

                # break case, we have run out of things to delete
                if not list(graph.neighbors(cur_node)):
                    break

                del_node = next(graph.neighbors(cur_node))
                # print(list(graph.neighbors(cur_node)))
                while del_node not in outside_pts[region_idx] or del_node in used_pts:
                    if del_node in used_pts and graph.degree(cur_node) == 1:
                        break
                    del_node = next(graph.neighbors(cur_node))
                    # cur_node = del_node
                    # print(del_node)


            potential_start_idxs.append(potential_start_idx)
            # print(potential_start_idxs)
            graph.remove_nodes_from(list(nx.isolates(graph)))
        # work through lattice and create a spanning tree vis DPS

        
        # print(potential_start_idxs)
        # nx.draw(graph, pos=posgen(graph), node_size = 1, with_labels=False)
        # return None, None
        
        # print(len(list(nx.connected_components(graph))))
        roots = []
        for start_idx in potential_start_idxs:

            if start_idx in list(graph.nodes):
                visited = set()  # Set to keep track of visited nodes
                tree_edges = []  # List to store edges of the spanning tree

                def dfs(node, parent):
                    # mark the current not as having been visited
                    visited.add(node)

                    # iterate through each connected node
                    for neighbor in graph.neighbors(node):
                        if neighbor != parent:

                            # found a node that hasn't been visited, so continue down path
                            if neighbor not in visited:
                                tree_edges.append((node, neighbor))
                                dfs(neighbor, node)

                            # Node has already been visited, so create leaf of tree
                            else:
                                # double check we haven't created this edge before
                                if (node, neighbor) not in tree_edges and (neighbor, node) not in tree_edges:
                                    tree_edges.append((node, neighbor))

                dfs(start_idx, None)
                
                # # create a dictionary to hold the nodes of the lattice
                unique_nodes = {}

                # # get unique nodes and insert into dictionary
                for edge in tree_edges:
                    node_1, node_2 = edge
                    if node_1 not in unique_nodes.keys():
                        unique_nodes[node_1] = TreeNode(node=node_1)
                    if node_2 not in unique_nodes.keys():
                        unique_nodes[node_2] = TreeNode(node=node_2)
                

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
                # print(potential_start_idx)
                
                root = unique_nodes[start_idx]
                # print(root)

                
            
                # self.plot_tree(root)  
                roots.append(root)
        
        return roots, graph

    
    def plot_tree(self, node: TreeNode) -> None:
        # print(node.node)
        for child in node.children:
            plt.plot([node.pos[0], child.pos[0]],[node.pos[1],child.pos[1]],"-k")
            self.plot_tree(child)
            

    def generate_path(self, tree: TreeNode) -> List[Tuple[float]]:

        points = []
        def dfs(node) -> None:
            
            points.append(node.pos)
            # print(points)

            # calc relative angles to next point
            rel_angles = []
            if len(node.children) > 1:
                if node.parent:
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





def intersection_point(line1, line2):
    """
    Calculate the intersection point of two lines.
    """

    p1 = np.array(line2[0])
    p2 = np.array(line2[1])
    p3 = np.array(line1[1])

    vec_1 = p2 - p1
    vec_2 = p3 - p1

    proj_dist =( vec_2 @ vec_1 ) / (vec_1 @ vec_1)

    proj_dist = max(0,proj_dist)

    proj_dist = min(proj_dist, np.linalg.norm(vec_1))
        

    point = proj_dist*vec_1 + p1

    return point[0], point[1]


def intersect_in_range(line1, line2):
    x,y = intersection_point(line1, line2)
    
    # Check if projected point lies on line2
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # print(x,y)

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




class Slicer:
    def __init__(self, params: Dict):
        """
        Initialize Slicer object with parameters dictionary
        Params: {
            "layer_height": float,
            "base_layers": int,
            "top_layers": int,
            "infill": "square" | "hexagonal"
        }
        """
        self.params = params

    def slice(self, path: str) -> None:
        """
        Take input model path and slice according to pre-set parameters
        """
        self.load_part(path)
        self.lattice = self.generate_lattice()

        # First layer generation
        self.layer_edges = self.create_raw_slices()
        

        # layer cleanup to create graphs
        if os.path.isfile("pickled-vars/poly.pckl"):
            f = open('pickled-vars/poly.pckl', 'rb')
            self.layer_polygons = pickle.load(f)
            f.close()
        else:
            self.layer_polygons = self.slice_to_polly(self.layer_edges)
            f = open('pickled-vars/poly.pckl', 'wb')
            pickle.dump(self.layer_polygons, f)
            f.close()

        # self.plot_mesh()
        # self.plot_layer_edge(3)
        # self.plot_lattice()

        if os.path.isfile("pickled-vars/graph.pckl"):
            f = open('pickled-vars/graph.pckl', 'rb')
            self.layer_graphs = pickle.load(f)
            f.close()
        else:
            self.layer_graphs = self.generate_layer_graphs(self.lattice, self.layer_polygons)
            f = open('pickled-vars/graph.pckl', 'wb')
            pickle.dump(self.layer_graphs, f)
            f.close()

        # self.plot_layer_graph(0)

        
        self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        if os.path.isfile("pickled-vars/path.pckl"):
            f = open('pickled-vars/path.pckl', 'rb')
            self.layer_paths = pickle.load(f)
            f.close()
        else:
            self.layer_paths, self.layer_full_graphs = self.planner.plan()            
            f = open('pickled-vars/path.pckl', 'wb')
            pickle.dump(self.layer_paths, f)
            f.close()
            f = open('pickled-vars/full_graphs.pckl','wb')
            pickle.dump(self.layer_full_graphs, f)
            f.close()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for layer_idx, layer_data in enumerate(self.layer_paths):
            for pts in layer_data:
                ax.plot(*pts.T, layer_idx*self.params["layer_height"])

        self.planner.generate_gcode(self.layer_paths)


        
    def load_part(self, path: str) -> None:
        """
        Preform all the necessary initializations when loading a model from a file
        """
        self.mesh = trimesh.load(path)
        self.mesh.rezero()
        self.bound = self.mesh.bounds
        self.x_range = self.bound[:,0]
        self.y_range = self.bound[:,1]
        self.z_range = self.bound[:,2]

    def plot_mesh(self) -> None:
        """
        Plot the model in a 3D pytplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_trisurf(self.mesh.vertices[:, 0],
                        self.mesh.vertices[:,1], 
                        triangles=self.mesh.faces, 
                        Z=self.mesh.vertices[:,2], 
                        alpha=1)
        
    def create_raw_slices(self) -> np.ndarray:
        """
        Take model and parameters and slice model uniformly along the xy axis. 
        Returns List[List[Polygon]]
        """

        layer_heights = np.arange(0, self.z_range[1], self.params["layer_height"])
        self.n_layers = np.size(layer_heights)
        layer_edges, _, _ = trimesh.intersections.mesh_multiplane(self.mesh, np.zeros((3)), np.array([0,0,1]), layer_heights)
        
        return layer_edges

    def slice_to_polly(self, layer_edges: np.ndarray) -> List[List[Polygon]]:
        """
        Convert the raw edge data into a set of polygons
        """

        pbar = tqdm(total=self.n_layers,desc = "Processing Layer Info")

        slices = []
        for layer in range(self.n_layers):
            # la/yer = 33
            re_ordered_edge = self.reorder_edges(layer_edges[layer])
            polygons = []
            for ring in re_ordered_edge:
                polygons.append(Polygon(ring,  edgecolor='b', facecolor='none'))
                # print(ring)

            slices.append(polygons)
            pbar.update(1)

            


            

            # print(slices)
            
        return slices
    



    def reorder_edges(self, coordinates) -> List[np.ndarray]:
        """
        Iterate through all sets of edges and reorder them. Also split into separate rings if needed.
        """
        used_points = []
        
        coordinates = np.around(coordinates,4).tolist()
        # coordinates = coordinates.tolist()
        reordered_coords = []
        
        while coordinates:
            current_edge = coordinates.pop(0)

            if current_edge[0] in used_points or current_edge[1] in used_points:
                break
                       
            reordered_region = [current_edge[0],current_edge[1]]

            while True:
                next_point_found = False

                for i, edge in enumerate(coordinates):
                    if edge[0] == reordered_region[-1]:
                        reordered_region.append(edge[1])
                        used_points.append(edge[1])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                    if edge[1] == reordered_region[-1]:
                        reordered_region.append(edge[0])
                        used_points.append(edge[0])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                
                if not next_point_found:
                    break
            
                # coordinates.pop(next_edge_index)
            
            # if reordered_region[0] != reordered_region[-1]:
            #     continue

            reordered_coords.append(np.array(reordered_region))
            
        
        # edge_rings = [ring[:, 0, :] for ring in reordered_coords]


        
        return reordered_coords
            
    def plot_layer_edge(self, layer: int) -> None:
        """
        Plot a given layer_edge
        Input: 
            layer: int
        """
        # plt.figure()

        layer_edge = self.layer_edges[layer]

        for idx in range(np.shape(layer_edge)[0]):
            plt.plot(*layer_edge[idx,:,:].T, "-k")

    def calc_n_regions_layer(self, layer) -> int:
        """
        Calculate the number of distinct closed regions for any given layer
        """
        return len(self.slice_polygons[layer])

    def is_in_layer(self, layer: int, point: Tuple[float,float]):
        """
        Calculate if a given point is within the model at a given layer
        """
        
        for poly in self.slice_polygons[layer]:
            if poly.get_path().contains_point(point):
                return True
            
        return False

    def generate_lattice(self) -> Lattice:
        """
        Generate a set lattice for the infill
        """
        size = self.params["infill_size"]
        infill_type = self.params["infill"]

        if infill_type == "square":
            latt = lp.Lattice.square(size)
            latt.add_atom()
            latt.add_connections()
        if infill_type == "triangle":
            latt = lp.Lattice.hexagonal(size)
            latt.add_atom()
            latt.add_connections()
        if infill_type == "hexagon":
            latt = lp.graphene(size)

        s = latt.build((self.x_range[1] + 2*size, self.y_range[1] + 2*size), pos = (-size,-size))

        return latt


    def generate_layer_graphs(self, lattice: lp.Lattice, layer_polygons: List[List[Polygon]]) -> List[List[nx.Graph]]:
        """
        Given the layer polygon and lattice, generate a networkx of the union
        """
    
        # update bar
        pbar = tqdm(total=self.n_layers,desc = "Converting Layer to Graph")
 

        layer_graphs = []
        for layer_idx in range(self.n_layers):
            polygons = self.layer_polygons[layer_idx]

            graphs = []
        
            for polygon in polygons:

                # get all unique lattice points and edges
                lat_points = []
                lat_edges = []

                for idx in range(lattice.num_sites):
                    lat_points.append(lattice.position(idx).tolist())
                    site_neighbors = lattice.neighbors(idx)
                    for neighbor in site_neighbors:
                        if neighbor > idx:
                            lat_edges.append([idx, neighbor])
                        else: 
                            lat_edges.append([neighbor, idx])

                lat_edges = np.unique(np.array(lat_edges),axis=0).tolist()

  
                
                poly_points = polygon.get_xy()

                # plt.plot(*poly_points.T)

                # buff_polygons = []


                buffer_poly = shp.Polygon(poly_points).buffer(-3*self.params["line_width"])

                poly_exteriors = []
                if type(buffer_poly) == shp.MultiPolygon:
                    for sub_poly_idx in range(len(buffer_poly.geoms)):
                        poly_exteriors.append(Polygon(np.array(buffer_poly.geoms[sub_poly_idx].exterior.coords.xy).T))
                else:
                    poly_exteriors.append(Polygon(np.array(buffer_poly.exterior.coords.xy).T))


                # for sub_poly in buff_polygons:

                
                


                # buffer_poly_points = np.array(buffer_poly.exterior.coords.xy).T
                
                # polygon = Polygon(buffer_poly_points)
                poly_points_array = [polygon.get_xy() for polygon in poly_exteriors]

                
                # iterate through all polygon edges and lattice edges to find intersecting sets
                problem_edges = defaultdict(list)

                # calc ahead of time all 
                for poly_points in poly_points_array:
                    for poly_point_idx in range(len(poly_points) - 1):
                        A = poly_points[poly_point_idx ,:]
                        B = poly_points[poly_point_idx + 1,:]
                        
                        # iterate through all edges in the lattice
                        for edge_idx, edge in enumerate(lat_edges):
                            C = lat_points[edge[0]]
                            D = lat_points[edge[1]]
                            
                            # see if polygon line AB intersects with lattice line CD
                            if intersect(A,B,C,D):

                                # calc where they intersected
                                line1 = LineString([A, B])
                                line2 = LineString([C, D])
                                int_pt = line1.intersection(line2)
                                int_coord = (int_pt.x, int_pt.y)

                                # label this edge as problematic
                                problem_edges[edge_idx].append(int_coord)

                # define a new set of points and lines. These will replace the lines that intersect. They are the trimmed lines



                # go through each problematic line
                critical_points = []
                for edge_idx, intersects in problem_edges.items():
                    
                    # get all the critical points in the problematic line in order
                    A = lat_points[lat_edges[edge_idx][0]]
                    B = lat_points[lat_edges[edge_idx][1]]

                    ordered_list = [A,B]

                    for coord in intersects:
                        ordered_list.append(list(coord))
                    
                    dist = lambda a : np.linalg.norm(np.array(A) - np.array(a))
                    ordered_list.sort(key=dist)

                    # calculate whether the sub line segment of problem line is in or outside of the polygon
                    mid_points = []
                    for idx in range(len(ordered_list) -1):
                        start = ordered_list[idx]
                        end = ordered_list[idx + 1]
                        mid_points.append([(start[0] + end[0])/2, (start[1] + end[1])/2])

                    # get the line segments that are valid
                    valid_mids = np.full((len(mid_points)), False)
                    for polygon in poly_exteriors:
                        valid_mids = valid_mids + polygon.get_path().contains_points(np.array(mid_points))
                    
                    # add the valid section of the problem line to the new points/new edges
                    for idx in range(len(ordered_list) -1):
                        start = ordered_list[idx]
                        end = ordered_list[idx + 1]

                        if valid_mids[idx]:
                            if start == A:
                                lat_points.append(end)
                                lat_edges.append([lat_edges[edge_idx][0], len(lat_points)-1])
                                critical_points.append(end)
                            elif end == B:
                                lat_points.append(start)
                                lat_edges.append([len(lat_points)-1, lat_edges[edge_idx][1]])
                                critical_points.append(start)
                            else:
                                lat_points.append(start)
                                lat_points.append(end)
                                lat_edges.append([len(lat_points)-2, len(lat_points)-1])
                                critical_points.append(start)
                                critical_points.append(end)
                
                # remove all the original problem edges
                cleaned_edges =[lat_edges[i] for i in range(len(lat_edges)) if i not in problem_edges.keys()]

                # create a list for our final edges
                final_edges = []

                # find all edges that connect to outside points
                inside_points = np.full((len(lat_points)), False)
                for polygon in poly_exteriors:
                    inside_points = inside_points + polygon.get_path().contains_points(np.array(lat_points))

                bad_points_idx = [idx for idx,val in enumerate(inside_points) if not val and lat_points[idx] not in critical_points]
                
                # only add edges to final edges if they do not touch outside points
                for edge in cleaned_edges:
                    if edge[0] not in bad_points_idx and edge[1] not in bad_points_idx:
                        final_edges.append(edge)


                # add valid lattice points to final points
                final_points = lat_points 

                # create graph
                G = nx.Graph()

                # create networkx vertex and edge data types
                graph_vertices = [(idx,{"x":val[0], "y":val[1]}) for idx,val in enumerate(final_points)]
                graph_edges = [tuple(i) for i in final_edges]

                G.add_nodes_from(graph_vertices)
                G.add_edges_from(graph_edges)

                # remove all isolated vertices
                G.remove_nodes_from(list(nx.isolates(G)))

                extra_edges = []
                for cur_node in list(G.nodes):
                    if G.degree(cur_node) == 1:
                        for other_node in list(G.nodes):
                            if G.degree(other_node) == 1 and cur_node != other_node:
                                x1 = G.nodes[cur_node]["x"]
                                y1 = G.nodes[cur_node]["y"]
                                x2 = G.nodes[other_node]["x"]
                                y2 = G.nodes[other_node]["y"]
                                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                                if dist < self.params["line_width"]:
                                    if (cur_node, other_node) not in extra_edges and (other_node, cur_node) not in extra_edges:
                                        # now check angle
                                        v1_neighbor = next(G.neighbors(cur_node))
                                        v2_neighbor = next(G.neighbors(other_node))
                                        if v1_neighbor != v2_neighbor:
                                            G.add_edges_from([(cur_node, other_node)])
                                            extra_edges.append((cur_node, other_node))
                        

                # nx.draw(G, pos=posgen(G),node_size = 1)

                # add graph to collection of graphs per layer
                G = G.to_undirected()
                graphs.append(G)
                
            layer_graphs.append(graphs)
            
            pbar.update(1)

        return layer_graphs
    
    def plot_layer_graph(self, layer: int) -> None:
        """
        Plot the networkx graph for a given layer
        """
        for G in self.layer_graphs[layer]:
            plt.figure()

            nx.draw(G, pos=posgen(G), node_size = 1, with_labels=True)

    def plot_lattice(self) -> None:
        latt = self.lattice
        latt.plot()
            
        


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)