"""
File for taking the graph and crating path and gcode
"""

import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from shapely import LineString
from shapely import buffer
import shapely as shp
from tqdm import tqdm
import sys

sys.setrecursionlimit(10000)

class TreeNode(object):
    """
    Tree style object for when creating a spanning tree over the graph. 
    """
    def __init__(self, pos: Tuple[float, float] = None, node: int = None, parent = None, children: List = None) -> None:
        """
        Initializes a single tree node

        Params:
            pos: Tuple[float,float], default None - position of the node in the tree
            node: int, default None - the networkx.Graph node number associated with this tree node
            parent: TreeNode, default None - the parent of the current node
            children : List[TreeNode], default [] - a list of children from the current node

        Returns: 
            None
        """
        self.node = node
        self.pos = pos
        self.parent = parent
        if children is None:
            self.children = []
        else:
            self.children = children
    
    def __hash__(self) -> int:
        """
        Creating a hashing for the TreeNode object so that we can put TreeNode into a dictionary
        """
        return hash(self.pos[0]) + hash(self.pos[1])

    def __eq__(self, other):
         """
         Create a way to compare two TreeNode objects to see if they are the same. Also so we can
         put them in a dictionary
         """
         return (
             self.__class__ == other.__class__ and
             self.pos == other.pos and
             self.parent == other.parent
        )
 

class Planner:
    """
    A class that takes the general structure of each layer as defined by the layer polygons and the
    lattice graph and generates a real path and gcode
    """
    def __init__(self, params: Dict, layer_polygons: List[List[Polygon]], layer_graphs: List[List[nx.Graph]]) -> None:
        """
        Initialize planner

        Params:
            params: Dict - a dictionary that contains all the necessary slicing and planning 
                parameters
            layer_polygons: List[List[Polygon]] - a list of lists of polygons that are the bounding
                regions of each layer
            layer_graphs: List[List[nx.Graphs]] - a list of lists of graphs that are the internal
                lattice of each region on each layer
        """
        self.layer_graphs = layer_graphs
        self.params = params
        self.layer_polygons = layer_polygons
        self.n_layers = len(layer_graphs)
    
    def generate_layer_paths(self) -> List[List[np.ndarray]]:
        """
        For all layers, generate spanning tree, then generate the final path

        Returns:
            layer_paths: List[List[np.ndarray]] - a list of list of numpy arrays that contain the xy
                coordinates of a valid path to print a layer
        """

        # create a little progress bar
        pbar = tqdm(total=self.n_layers,desc = "Planning Layer Path")

        # create a list to hold the pathing data for each layer
        layer_paths = []

        for layer_idx in range(self.n_layers):

            # create a list to hold the pathing data for each subregion in the layer
            region_paths = []
            
            # enumerate over each separate region in the layer
            for region_idx, graph in enumerate(self.layer_graphs[layer_idx]):
                polygon = self.layer_polygons[layer_idx][region_idx]

                # generate a spanning tree (or multiple) to cover a region in the layer
                trees =self.generate_spanning_tree(polygon, graph)

                # if a tree can cover this region, generate a path
                if trees:
                    for root in trees:
                        # using the tree, traverse over the tree using DFS to create a "point path"
                        point_path = self.generate_path(root)
                    
                        # create a line string from this point path
                        line = LineString(point_path)
                        
                        # create an offset from the LineString to create the final path
                        offset_path = np.array(line.buffer(.5*self.params["line_width"]).exterior.coords)

                        # save the final path into our regional_paths list
                        region_paths.append(offset_path)

            layer_paths.append(region_paths)

            # update our progress bar after finishing 1 layer
            pbar.update(1)
        
        # return the layer paths for all layers
        return layer_paths

    def generate_gcode(self, layer_paths: List[List[np.ndarray]]):
        """
        Generate a gcode file that a 3D printer could use to print the input geometry.
        
        params:
            layer_paths: List[List[np.ndarray]] - a list of list of numpy arrays that contain the xy
                coordinates of a valid path to print a layer
        """

        # create a little progress bar
        pbar = tqdm(total=self.n_layers,desc = "Generating gcode")

        # keep a global counter of how much we have extruded and our current layer number
        extrusion = 0
        layer_num = 0 

        def generate_gcode_for_layer(layer_loops: List[np.ndarray], extrusion: float, layer_num: int, feed_rate=100, retract_distance=.08, lift_distance=2) -> Tuple[List, float, int]:
            """
            Each layer is made up of several loops. For each individual loop on a layer, write gcode
            to print that loop, then retract filament, move up, travel to the next loop, and repeat

            Params:
                layer_loops: List[np.ndarray] - list of nx3 arrays where each array is a separate
                    loop to print
                extrusion: float - the total amount the "4th axis" extrusion has gone through so far
                layer_num: int - the current layer number

            Returns:
                gcode: a list which contains the current gcode information
                extrusion: float - the total amount the "4th axis" extrusion has gone through so far
                layer_num: int - the current layer number
            """

            # create a list which will contain all the gcode statements
            gcode = []
            gcode.append(f";LAYER:{layer_num}")

            # iterate through each of the loops
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
                        # calculate the length of filament needed to print a single mm length section
                        ratio = self.params["layer_height"]*self.params["line_width"] / ((1.75/2)**2*3.14)

                        # calculate the filament needed to extrude from the current position to the next spot
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

        def generate_gcode_for_layers(layer_data: List[List[np.ndarray]], extrusion: float, layer_num: int, feed_rate=100):
            # global extrusion
            gcode = []
            
            # Initialize
            gcode.append("M82") # set absolute extrusion mode
            gcode.append("G21")  # Set units to millimeters
            gcode.append("G90")  # Set to absolute positioning
            gcode.append("G92 E0")  # Zero the extruder
            gcode.append("M82") # set absolute extrusion mode

            # set temps
            gcode.append("M104 S200") # nozzle
            gcode.append("M140 S60") # bed
            gcode.append("M190 S60") # bed
            gcode.append("M109 S200") # nozzle
            
            # Home
            gcode.append("G28")  # Home all axes
            gcode.append("G92 E0.0")
            gcode.append(f"G1 F2100 E{extrusion - .08}")
            
            # Set feed rates
            gcode.append(f"G1 F{feed_rate}")  # Set X, Y, and Z feed rate
            
            # Generate G-code for each loop
            gcode.append(f";LAYER_COUNT:{self.n_layers - 1}")
            for layer_loops in layer_data:
                new_code, extrusion, layer_num = generate_gcode_for_layer(layer_loops, extrusion, layer_num)
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
            gcode.append("M30") 
            
            return "\n".join(gcode)
        
        layer_data = []

        for layer_idx, layer in enumerate(layer_paths):
            layer_loops = []
            layer_height = layer_idx * .2
            for array in layer:
                new_column = np.full((array.shape[0], 1), layer_height)

                # horizontally stack arrays to give z layer height to points
                new_arr = np.hstack((array, new_column))
                layer_loops.append(new_arr)
            
            layer_data.append(layer_loops)
            



        # Generate G-code
        gcode_content = generate_gcode_for_layers(layer_data, extrusion, layer_num)

        # Save G-code to a file
        with open("output.gcode", "w") as file:
            file.write(gcode_content)



    def generate_spanning_tree(self, polygon: Polygon, graph: nx.Graph) -> List[TreeNode]:
        """
        Given the bounding polygon and infill, generate a tree that first covers all the outside edges, then then inside

        Params:
            polygon: Polygon - a bounding polygon for a region in a layer
            graph: nx.Graph - a lattice graph for a region in a layer

        Returns:
            roots: Lit[TreeNode] - a list of tree root nodes that fully cover a whole layer 
        """


        """ ---------------------------------------------
        Step 1) Create a buffered polygon exterior
        -------------------------------------------------"""


        # get buffer polygon that is 1x width smaller than original polygon, this is when we print
        # perimeter of our part, our part remains dimensionally accurate
        poly_points = polygon.get_xy()
        buffer_poly = shp.Polygon(poly_points).buffer(-self.params["line_width"])

        # now we have to deal with the possible edge case that if we buffer the whole exterior by a 
        # line width, it is possible that our single polygons splits into multiple polygons, so now
        # we create a list of polygons to deal with, if our polygon is still only one region, our
        # list will only have one item in it.
        poly_exteriors = []
        if type(buffer_poly) == shp.MultiPolygon:
            for sub_poly_idx in range(len(buffer_poly.geoms)):
                poly_exteriors.append(Polygon(np.array(buffer_poly.geoms[sub_poly_idx].exterior.coords.xy).T))
        else:
            poly_exteriors.append(Polygon(np.array(buffer_poly.exterior.coords.xy).T))

        # get a numpy array for each possible polygon region
        poly_points_array = [polygon.get_xy() for polygon in poly_exteriors]

        # if any of the regions are null, return None
        # NOTE: I think this might be legacy code and isn't needed anymore
        for poly_points in poly_points_array:
            if not np.any(poly_points):
                return None, None
            

  
        """ ---------------------------------------------
        Step 2) Connect the lattice to the outside buffered polygon
        -------------------------------------------------"""

                    
        # get list of connected graphs
        connected_components = list(nx.connected_components(graph))

        # for each subgraph, find an end piece
        end_nodes = []

        # for each end node, add an edge that connects them to the outside
        new_end_nodes = []
        insert_idxs = [(0,0) for _ in connected_components]

        def find_connection(region_idx: int, node: int):
            """
            Given a node in a graph, find the best way to connect it to the outside bounding polygon
            """

            prev_node = next(graph.neighbors(node))
            
            # get line that extends outwards
            line_1 = [(graph.nodes[prev_node]["x"],graph.nodes[prev_node]["y"]), 
                      (graph.nodes[node]["x"],graph.nodes[node]["y"])]
            
            best_dist = math.inf
            best_point = (None,None)
            idx_total = 0

            # iterate through all edges in all polygon regions
            for poly_points in poly_points_array:
                for idx in range(len(poly_points) - 1): 

                    # get the edge of polygon as line 2   
                    line_2 = [tuple(poly_points[idx]), tuple(poly_points[idx+1])]

                    # find best intersection point between line 1 and line 2
                    point = intersection_point(line_1, line_2)
                    x,y = point

                    # compare this connection to other connections already found
                    dist = math.sqrt((line_1[1][0] - x)**2 + (line_1[1][1] - y)**2)
                    if dist < best_dist:
                        best_point = point
                        best_dist = dist

                        # if this is the best we have found so far, save the idexes
                        # NOTE: I don't really recall what the following logic is for, but it has to
                        # do with some edge cases handling a split regions
                        if idx_total + idx+1 == len(poly_points) - 1:
                            insert_idxs[region_idx] = (idx+idx_total, idx_total)
                        else:
                            insert_idxs[region_idx] = (idx+idx_total, idx_total + idx+1)
                idx_total += idx

            # if we really cannot find a way to connect, return None
            # NOTE: I don't think this bit of code is used anymore, could probably delete?
            if abs(best_dist - 2*self.params["line_width"]) > .01:
                return (None, None)

            return best_point
            
        # for each section of the graph, find the connection point and add it to the new end nodes list
        for region_idx, sub_graph in enumerate(connected_components):
            for node in sub_graph:
                if graph.degree(node) == 1:
                    connection_pt = find_connection(region_idx, node)
                    
                    # if the connection is note (None, None), save point and break
                    if not connection_pt[0] is None:
                        end_nodes.append(node)
                        new_end_nodes.append(connection_pt)
                        break

        # get the highest node number in the graph or zero if there is no graph/lattice
        if connected_components:
            graph_n_nodes = max(graph.nodes)
        else:
            graph_n_nodes = 0


        # iterate through all the points in all the possible polygon regions and add their points
        # to the list new_points
        idx = 0
        new_points = []
        for poly_points in poly_points_array:            
            for p_idx in range(len(poly_points)-1):
                x1 = poly_points[p_idx][0]
                y1 = poly_points[p_idx][1]
                new_points.append((idx + graph_n_nodes + 1,{'x':x1, 'y':y1}))
                idx += 1

        # also add in the new end nodes that connect the internal lattice to the outside
        for new_end_node_idx, new_end_node in enumerate(new_end_nodes):
            x1, y1 = new_end_node
            new_points.append((idx + graph_n_nodes + new_end_node_idx + 1, {"x":x1, "y":y1}))
            
        # add all the new points to the graph
        graph.add_nodes_from(new_points)
        
        # create a list to keep track of the node numbers for all exterior nodes in the graph
        outside_pts = []

        # create a list that will store tupes of node idx that will be edges in the graph
        new_edges = []

        vert_idx = 0
        sub_offset = 0
        # iterate through  the possible polygon regions
        for poly_points in poly_points_array:
            regional_outside_pts = []
            # iterate through all but the last two points (because the last one is a duplicate of the first)
            for edge_idx in range(len(poly_points) - 2):
                # calculate the node number the points that form the edge
                node_1_number = vert_idx+graph_n_nodes+1+sub_offset
                node_2_number = vert_idx+graph_n_nodes+2+sub_offset

                # add the node numbers to the regional outside pts list
                regional_outside_pts.append(node_1_number)
                regional_outside_pts.append(node_2_number)  

                # if the current edge is not where we are connecting the internal lattice to, just
                # add the edge to the new_edges list         
                if (vert_idx , vert_idx+1) not in insert_idxs and (vert_idx + 1, vert_idx) not in insert_idxs:
                    new_edges.append((node_1_number, node_2_number))

                # if this is where we are connecting the lattice to the exterior, we need to do a
                # of work to get everything sorted
                else:
                    sub_graph_idx = insert_idxs.index((vert_idx, vert_idx+1))
                    # calculate the node number for the new connection node
                    node_3_number = graph_n_nodes + idx + sub_graph_idx + 1     
                    new_edges.append((node_1_number, node_3_number))
                    new_edges.append((node_2_number, node_3_number))
                    new_edges.append((end_nodes[sub_graph_idx], node_3_number))
                    regional_outside_pts.append(node_3_number)
                vert_idx +=1
            
            # again do the same process for the very last edge that wasn't covered by our enumeration

            # again re-calculate our node numbers
            node_1_number = vert_idx+graph_n_nodes+1+sub_offset
            node_2_number = vert_idx+graph_n_nodes+3+sub_offset - len(poly_points)
            node_3_number = graph_n_nodes + idx + sub_graph_idx + 1   

            # calculate to see if we are performing a connection on the last edge or not
            last_edge = ( vert_idx, vert_idx + 2 - len(poly_points))
            if last_edge not in insert_idxs and (last_edge[1],last_edge[0]) not in insert_idxs:
                # if not, just add the edge as normal
                new_edges.append((node_1_number, node_2_number))
            else:
                # if we are performing a connection, do so
                sub_graph_idx = insert_idxs.index(last_edge)
                new_edges.append((node_1_number, node_3_number))
                new_edges.append((node_3_number, node_2_number))
                new_edges.append((node_3_number, end_nodes[sub_graph_idx]))

            sub_offset += 1
            outside_pts.append(regional_outside_pts)

        # add all the new edges
        graph.add_edges_from(new_edges)


        """ ---------------------------------------------
        Step 3) Create a small gap on the outside of the polygon
        -------------------------------------------------"""

        # we are creating a small gap on the outside of the polygon to create an "opening" that will
        # serve as the entry point for the tree. We are going to first try looking at the spots 
        # were we connected the lattice to the outside as a starting point
       
        # creating a list of points we've already looked over incase we try to delete the same point
        # twice. NOTE: might be unnecessary, could maybe delete
        used_pts = []

        # gets a valid starting point to start deleting edges. Needs to be on outside and connect to
        # inside
        def get_valid_start_pt(outside_pts):
            for idx in outside_pts:
                if graph.degree(idx) == 3:
                    return idx
            return idx
        
        potential_start_idxs = []

        potential_start_idx = graph_n_nodes
        # iterate through each region in the layer
        for region_idx, poly_points in enumerate(poly_points_array):
            # get initial starting point
            potential_start_idx = get_valid_start_pt(outside_pts[region_idx])

            # get the next node on the outside of the graph
            del_node = next(graph.neighbors(potential_start_idx))
            while del_node not in outside_pts[region_idx]:
                del_node = next(graph.neighbors(potential_start_idx))
            
            cur_node = potential_start_idx
            used_pts.append(cur_node)

            # keep track of the original starting point
            x_1 = graph.nodes[potential_start_idx]["x"]
            y_1 = graph.nodes[potential_start_idx]["y"]
            
            # iteratively delete edges until we reach the break condition
            while True:

                # get the location of teh current end point
                x_2 = graph.nodes[del_node]["x"]
                y_2 = graph.nodes[del_node]["y"]
                

                # Delete edges
                graph.remove_edges_from([(del_node, cur_node)])

                # if we ever create a gap that is greater than 2x the line_width, stop, and fix the 
                # gap so that it is exactly 2x the line width gap
                dist = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                if dist > 2*self.params["line_width"]:
                    
                    # calculate the position of a new point to add to the exterior edge
                    vec_a = np.array([x_1,y_1]) - np.array([x_2, y_2])
                    vec_a = vec_a/np.linalg.norm(vec_a) * (dist - 2*self.params["line_width"])
                    new_point = (x_2 + vec_a[0], y_2 + vec_a[1])
                    
                    # add the new node and edge to the graph
                    graph.add_nodes_from([(max(graph.nodes) + 1, {"x": new_point[0], "y":new_point[1]})])
                    graph.add_edges_from([(del_node, max(graph.nodes))])
                    
                    break

                # if we did not reach the break case, keep moving along the edge
                cur_node = del_node

                # break case, we have run out of things to delete
                if not list(graph.neighbors(cur_node)):
                    break

                # find the next del node. Make sure it's an exterior point and not already used
                del_node = next(graph.neighbors(cur_node))
                while del_node not in outside_pts[region_idx] or del_node in used_pts:
                    if del_node in used_pts and graph.degree(cur_node) == 1:
                        break
                    del_node = next(graph.neighbors(cur_node))


            # save our start index into a list. NOTE: the name "potential_start_idxs" is a slight 
            # misnomer since I don't think it's potential anymore, its the actual start idx. But I 
            # don't want to change that now.
            potential_start_idxs.append(potential_start_idx)

            # clean up any remaining junk in the graph
            graph.remove_nodes_from(list(nx.isolates(graph)))


        """ ---------------------------------------------
        Step 4) DFS through the graph to make a tree
        -------------------------------------------------"""

        # The way we are doing DFS through the graph to make a tree is to first DFS to create a list
        # of edges. The list of edges just tells us how things are connected, but does not create a 
        # tree yet. Then after we create this list of edges, we iterate through the edges and form
        # the tree. It's an unintuitive 2 step process that can probably be simplified into a single
        # step. But that's a later problem.

        # work through lattice and create a spanning tree vis DPS
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
                
                # create a dictionary to hold the nodes of the lattice
                unique_nodes = {}

                # get unique nodes and insert into dictionary
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

                        # since our leaf cannot directly overlap the previous node at that location,
                        # we are leaving a small gap between them, depending on the shape of the lattice.
                        # we are making the offset only dependent on the lattice shape because we make
                        # the lattice only ever connect with the outside at a minimal number of spots
                        # so if there is an intersection, we know it must be internally to the lattice
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
                
                root = unique_nodes[start_idx]

                roots.append(root)
        
        return roots


    def generate_path(self, tree: TreeNode) -> List[Tuple[float]]:
        """
        Given the root of a tree, generate a path around the tree using DFS

        Params:
            tree: TreeNode - the root of the tree
        """

        # create a list to hold all the points of our DFS traversal
        points = []

        def dfs(node) -> None:
            points.append(node.pos)

            # calc relative angles to next point. We want to go through each of the children in order
            # of relative angles in order to ensure that we are creating non-self-intersecting paths
            rel_angles = []
            if len(node.children) > 1:
                if node.parent:
                    for child in node.children:
                        # create vectors to define the positions of each point
                        v_0 = np.array(node.pos).reshape((2, 1))
                        v_1 = np.array(node.parent.pos).reshape((2, 1))
                        v_2 = np.array(child.pos).reshape((2, 1))
                        
                        # calculate the relative angle and add it to the rel_angles list
                        rel_angles.append(angle(v_0,v_1,v_2)) 

                    # reorder children using the rel_angles as a key
                    node.children = [x for _, x in sorted(zip(rel_angles, node.children), key=lambda pair: pair[0])]

            # now that the children are re-sorted by relative angle, just go through them recursively
            for child in node.children:
                dfs(child)

            # after we have searched through all the nodes, also add the parent position to the list
            # to simulate us moving back up the tree during our traversal
            if node.parent:
                points.append(node.parent.pos)

        dfs(tree)
        
        return points


    def plot_tree(self, node: TreeNode) -> None:
        """
        Function to plot a tree only given it's initial root node, recursively go through tree plotting
        tree branches until we reach leaf nodes.

        Params:
            node: TreeNode - the root node of the tree to start plotting at
        """
        for child in node.children:
            plt.plot([node.pos[0], child.pos[0]],[node.pos[1],child.pos[1]],"-k")
            self.plot_tree(child)
            


def intersection_point(line1, line2):
    """
    Calculate the intersection point of two lines, or at least the closest point
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
