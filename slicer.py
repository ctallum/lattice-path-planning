"""
File to contain Slicer class which takes a stl model and generates cross sections with different
infill geometry
"""

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
import math

from planner import Planner

class Slicer:
    def __init__(self, params: Dict) -> None:
        """
        Initialize Slicer object with parameters dictionary
        
        Params:
            params: dict - A dictionary that contains the following set values: 
                "layer_height": float, "infill": "square" | "triangle" | "hexagon", and
                "line_width": float
        """
        self.params = params

    def slice(self, path: str, debug_mode = False) -> None:
        """
        Take an input path str to a 3D object file. Take the file and slice it according go given
        parameters, and export path to a gcode file. The slicer first converts model into a set of
        layers, then it converts each layer into an undirected graph. It then creates a tree over
        the graph, and performs a DFS traversal to create a single, non-intersecting path.

        Params:
            path: str - a string that leads to the 3D model file
            debug_mode: bool - an option that causes the slicer to save different
                variables as pickle file during the running of the code. When debugging the code, it
                saves time re-running the same section of code over and over

        Returns:
            None
        """

        # if we are in debug mode, jump to debug function
        if debug_mode:
            return self.slice_debug(path)
        
        # if we are not in debug mode, continue slicing as normal

        # load the mesh and slice and perform a quick generation of the internal lattice 
        self.load_part(path)
        self.lattice = self.generate_lattice()
        self.layer_edges = self.create_raw_slices()

        # convert each slice of the 3D model into a convenient polygon object
        self.layer_polygons = self.slice_to_polly()

        # using the lattice and layer polygon, generate a graph for each layer
        self.layer_graphs = self.generate_layer_graphs()

        # load in our planner object to do the actual path planning on the graph
        self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        # generate the layer paths
        self.layer_paths = self.planner.generate_layer_paths()            

        # generate a gcode file
        self.planner.generate_gcode(self.layer_paths)

        # plot the finals paths
        self.plot_final_paths()

    def slice_debug(self, path: str) -> None:
        """
        Perform the exact same operations as the main slice function, but in order to save time in
        debugging and developing code, save major variables as pickle files and load them instead of
        re-running the whole code. 
        """

        # load the mesh and slice and perform a quick generation of the internal lattice 
        self.load_part(path)
        self.lattice = self.generate_lattice()
        self.layer_edges = self.create_raw_slices()

        # base directory for all pickled variables
        base_dir = "pickled-vars/"

        # extra set of variables to force any section to re-run, even if pickle files exist
        force_slice_to_poly = False
        force_generate_layer_graphs = False
        force_generate_layer_paths = False

        # check to see if layer polygons have already been created
        layer_poly_file = base_dir + "layer_polygons.pckl"
        if os.path.isfile(layer_poly_file) and not force_slice_to_poly:
            with open(layer_poly_file, 'rb') as f:
                self.layer_polygons = pickle.load(f)
        else:
            self.layer_polygons = self.slice_to_polly()
            with open(layer_poly_file, 'wb') as f:
                pickle.dump(self.layer_polygons, f)

        # check to see if layer graphs have already been created
        layer_graphs_file = base_dir + "layer_graphs.pckl" 
        if os.path.isfile(layer_graphs_file) and not force_generate_layer_graphs:
            with open(layer_graphs_file, 'rb') as f:
                self.layer_graphs = pickle.load(f)
        else:
            self.layer_graphs = self.generate_layer_graphs()
            with open(layer_graphs_file, 'wb') as f:
                pickle.dump(self.layer_graphs, f)
        
        # pull in the separate planner object to do the final path planning
        self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        # check to see if layer_paths have already been created
        layer_paths_file = base_dir + "layer_paths.pckl"
        if os.path.isfile(layer_paths_file) and not force_generate_layer_paths:
            with open(layer_paths_file, 'rb') as f:
                self.layer_paths = pickle.load(f)
        else:
            self.layer_paths = self.planner.generate_layer_paths()            
            with open(layer_paths_file, 'wb') as f:
                pickle.dump(self.layer_paths, f)

        # generate a gcode file
        self.planner.generate_gcode(self.layer_paths)
        
        # plot final paths
        self.plot_final_paths()        
        
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

        # build a lattice that is slightly larger than x_range and y_range of the model.
        latt.build((self.x_range[1] + 2*size, self.y_range[1] + 2*size), 
                   pos = (-size, -size))

        return latt
        
    def create_raw_slices(self) -> np.ndarray:
        """
        Take model and parameters and slice model uniformly along the xy axis. 
        
        Returns:
            np.ndarray
        """

        # create a vector that contains the height of each layer
        layer_heights = np.arange(0, self.z_range[1], self.params["layer_height"])
        self.n_layers = np.size(layer_heights)

        # create plane origin and normal vector to plane
        origin = np.zeros((3))
        normal = np.array([0,0,1])

        # create slices starting at the origin, in the direction of the normal with
        # layer height specified
        layer_edges, _, _ = trimesh.intersections.mesh_multiplane(self.mesh, origin, 
                                                                  normal, layer_heights)
        
        return layer_edges

    def slice_to_polly(self) -> List[List[Polygon]]:
        """
        Convert the raw edge data into a set of polygons. They are originally passed as a set of
        line segments, but the line segments may not be ordered. So first order the line segments,
        then convert them into a matplotlib.patches Polygon object

        Returns:
            List[List[Polygon]]
        """

        # create a little progress bar
        pbar = tqdm(total=self.n_layers,desc = "Processing Layer Info")

        # slices will represent each layer's set of polygons
        slices = []
        for layer in range(self.n_layers):
            
            # reorder all the edge data, get back a list of distinct regions
            ordered_regions = self.reorder_edges(self.layer_edges[layer])
            
            # convert each region in each layer to its own polygon object
            polygons = []
            for ring in ordered_regions:
                polygons.append(Polygon(ring,  edgecolor='b', facecolor='none'))

            slices.append(polygons)
            
            # update the progress bar
            pbar.update(1)
            
        return slices

    def reorder_edges(self, coordinates: np.ndarray) -> List[np.ndarray]:
        """
        Iterate through all sets of edges and reorder them. Also split into separate rings if needed
        """

        used_points = []
        
        # first round all the coordinates to 4 digits long. There was a bug in which two points 
        # which should have been the same ones had a very slight different position value due to
        # rounding error. So fix this by rounding everything to 4. also convert numpy array to a 
        # list
        coordinates = np.around(coordinates,4).tolist()

        # create a blank list to layer store ordered regions
        ordered_regions = []
        
        # we are going to iterate over all the coordinates until we have added everything to a 
        # ordered region
        while coordinates:

            # get current edge
            current_edge = coordinates.pop(0)

            # disregard the edge if it has already been used, break out of loop we probably have
            # completed the loop or we have extra edges
            if current_edge[0] in used_points or current_edge[1] in used_points:
                break
                       
            # start a new re-ordered region using the current edge
            reordered_region = [current_edge[0],current_edge[1]]

            while True:

                # see if we can find a point that follows the current reordered region
                next_point_found = False

                for i, edge in enumerate(coordinates):
                    # if we find an edge where the first value is the same as the last value in our
                    # chain, add the second value to the ordered region
                    if edge[0] == reordered_region[-1]:
                        reordered_region.append(edge[1])
                        used_points.append(edge[1])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                    # if we find an edge where the second value is the same as the last value in the
                    # chain, add the first value to the ordered region
                    if edge[1] == reordered_region[-1]:
                        reordered_region.append(edge[0])
                        used_points.append(edge[0])
                        coordinates.pop(i)
                        next_point_found = True
                        break
                
                # if we cannot find the next value, then the loop has been completed
                if not next_point_found:
                    break
            
            # by now, we have a fully defined ordered region, add it to the list of ordered regions
            # per layer
            ordered_regions.append(np.array(reordered_region))
            
        return ordered_regions


    def generate_layer_graphs(self) -> List[List[nx.Graph]]:
        """
        Given the layer polygon and lattice, generate a networkx of the union

        Params:
            lattice: lp.Lattice
            layer_polygons: List[List[Polygon]] - A list of list of polygon objects
        """
    
        # create a little progress bar
        pbar = tqdm(total=self.n_layers,desc = "Converting Layer to Graph")
 
        # create a final list to hold all the layer graphs
        layer_graphs = []

        # iterate through each layer
        for layer_idx in range(self.n_layers):

            # get a list of polygons for each layer
            polygons = self.layer_polygons[layer_idx]

            # create the list of graphs for each layer. There may be multiple graphs for a given
            # layer because sometimes a model will split into multiple regions as the z height
            # changes. So as the model splits, we have different regions on the same layer
            graphs = []
        
            # iterate through each separate polygon on the same layer
            for polygon in polygons:

                # get all unique lattice points and edges
                lat_points = []
                lat_edges = []

                # get a list of all edges in the lattice
                for cur_node in range(self.lattice.num_sites):
                    lat_points.append(self.lattice.position(cur_node).tolist())
                    # insert the edge indices of the lattice into the list 
                    for neighbor in self.lattice.neighbors(cur_node):
                        lat_edges.append([cur_node, neighbor])

                # get all the unique edges
                lat_edges = np.unique(np.array(lat_edges),axis=0).tolist()

                # get the raw xy data
                poly_points = polygon.get_xy()

                # we are now creating a buffer polygon. This is a polygon tht will be smaller than
                # desired polygon by the width of 3 "line_widths". We are doing this for a few
                # reasons: to account for the fact that the final path will be slightly smaller than
                # the final shape to account for nozzle width. secondly, we will trim the lattice so
                # that it does not touch the polygon wall. This will save time later when generating
                # the final graph and tree
                buffer_poly = shp.Polygon(poly_points).buffer(-3*self.params["line_width"])

                # annoyingly, this buffer polygon may accidentally split into more than one polygon
                # this can happen if there is a thin point (like a figure 8). When that happens, we
                # have to treat it as a different region
                regional_polygons = []
                if type(buffer_poly) == shp.MultiPolygon:
                    for sub_poly_idx in range(len(buffer_poly.geoms)):
                        sub_polygon = Polygon(np.array(buffer_poly.geoms[sub_poly_idx].exterior.coords.xy).T)
                        regional_polygons.append(sub_polygon)
                else:
                    regional_polygons.append(Polygon(np.array(buffer_poly.exterior.coords.xy).T))

                # get all the exterior points of the polygons
                poly_points_array = [polygon.get_xy() for polygon in regional_polygons]

                
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

                # For each lattice line that intersected with the regional polygon, we are going to 
                # make a new lattice line that stops short, it is trimmed. 


                # go through each problematic line
                critical_points = []
                for edge_idx, intersection_pts in problem_edges.items():
                    
                    # get all the critical points in the problematic line in order
                    A = lat_points[lat_edges[edge_idx][0]]
                    B = lat_points[lat_edges[edge_idx][1]]

                    ordered_list = [A,B]

                    for coord in intersection_pts:
                        ordered_list.append(list(coord))
                    
                    dist = lambda a : np.linalg.norm(np.array(A) - np.array(a))
                    ordered_list.sort(key=dist)

                    # calculate whether the sub line segment of problem line is in or outside of the
                    # regional polygon
                    mid_points = []
                    for idx in range(len(ordered_list) -1):
                        start = ordered_list[idx]
                        end = ordered_list[idx + 1]
                        mid_points.append([(start[0] + end[0])/2, (start[1] + end[1])/2])

                    # get the line segments that are valid
                    valid_mids = np.full((len(mid_points)), False)
                    for polygon in regional_polygons:
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
                cleaned_edges = [lat_edges[i] for i in range(len(lat_edges)) if i not in problem_edges.keys()]

                # create a list for our final edges
                final_edges = []

                # find all edges that connect to outside points
                # create a boolean array initialized to all False
                inside_points = np.full((len(lat_points)), False)
                for polygon in regional_polygons:
                    # If any of the regional polygon contains the lat_point, keep them
                    inside_points = inside_points + polygon.get_path().contains_points(np.array(lat_points))

                # all the bad points are the ones that are not inside the regional polygon and are 
                # not critical points
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

                # construct the graph!
                G.add_nodes_from(graph_vertices)
                G.add_edges_from(graph_edges)

                # remove all isolated vertices
                G.remove_nodes_from(list(nx.isolates(G)))

                # weird edge case that we need to handle. Under certain conditions, when the lattice
                # is trimmed by the regional polygon (the buffered one), it can leave behind some
                # annoying geometry. Particularly if the lattice is triangle and sometimes hexagon. 
                # essentially, if we are have with a >| shape where the end of two lattice points are
                # facing into echoer (but are not quite touching), but are less than the param
                # line width, when later creating the path, it will jump the small gap, causing some
                # sections of the graph to not have a path. The solution is to bridge these small 
                # gaps ahead of time. 
                
                # store all our extra edges that we will add
                extra_edges = []

                # iterate through all end nodes in the lattice
                for cur_node in list(G.nodes):
                    if G.degree(cur_node) == 1:
                        for other_node in list(G.nodes):
                            if G.degree(other_node) == 1 and cur_node != other_node:
                                
                                # calculate the distance between the two nodes and see if it is too
                                # small
                                dist = dist_btw_graph_nodes(G, cur_node, other_node)
                                if dist < self.params["line_width"]:
                                    if (cur_node, other_node) not in extra_edges and (other_node, cur_node) not in extra_edges:
                                        
                                        # if they are too close, we need to also check if they have
                                        # incoming angles or outgoing angles. Incoming is like >|
                                        # while outgoing is like <|. Only incoming angles cause a 
                                        # problem. Can check by seeing if they share the same
                                        # neighbor.
                                        v1_neighbor = next(G.neighbors(cur_node))
                                        v2_neighbor = next(G.neighbors(other_node))
                                        if v1_neighbor != v2_neighbor:
                                            extra_edges.append((cur_node, other_node))

                # add in all our new edges
                G.add_edges_from(extra_edges)

                # add graph to collection of graphs per layer
                G = G.to_undirected()

                # add to the list of graphs per layer
                graphs.append(G)
                
            # add the layer graph to the list of all layers' graphs
            layer_graphs.append(graphs)
            
            # update our progress bar by 1
            pbar.update(1)

        return layer_graphs
    
    def plot_mesh(self) -> None:
        """
        Plot the model in a 3D pyplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_trisurf(self.mesh.vertices[:, 0],
                        self.mesh.vertices[:,1], 
                        triangles=self.mesh.faces, 
                        Z=self.mesh.vertices[:,2], 
                        alpha=1)
    
    def plot_lattice(self) -> None:
        """
        Plot the lattice
        """
        latt = self.lattice
        latt.plot()

    def plot_layer_edge(self, layer: int) -> None:
        """
        Plot a given layer_edge
        Params: 
            layer: int
        """
        layer_edge = self.layer_edges[layer]

        for idx in range(np.shape(layer_edge)[0]):
            plt.plot(*layer_edge[idx,:,:].T, "-k")

    def plot_layer_graph(self, layer: int) -> None:
        """
        Plot the networkx graph for a given layer
        """
        for G in self.layer_graphs[layer]:
            nx.draw(G, pos=posgen(G), node_size = 1, with_labels=True)

    def plot_final_paths(self) -> None:
        """
        Using the generated self.layer_paths, plot them all on a 3D matplotlib figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for layer_idx, layer_data in enumerate(self.layer_paths):
            for pts in layer_data:
                ax.plot(*pts.T, layer_idx*self.params["layer_height"])

# MISC FUNCTIONS

def dist_btw_graph_nodes(graph: nx.Graph, node_1: int, node_2: int) -> float:
    """
    Given a graph, calculate the distance between two nodes
    """
    x1 = graph.nodes[node_1]["x"]
    y1 = graph.nodes[node_1]["y"]
    x2 = graph.nodes[node_2]["x"]
    y2 = graph.nodes[node_2]["y"]

    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Return true if line segments AB and CD intersect
def intersect(A: List[List], B: List[List], C: List[List], D: List[List]):
    """
    Return true if line segments AB and CD intersect

    params:
        A,B,C,D: List[List] of coordinate points
    """
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def posgen(G: nx.Graph):
    """
    Returns a dictionary where for each key which is node in graph, we get the coordinate value of
    its position
    """
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
