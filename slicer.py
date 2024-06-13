"""
File to contain Slicer class which takes a stl model and generates cross sections with different
infill geometry
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, List
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

class Params:
    """
    A class to hold all slicing parameters

    Parameters:
        model_path [str] - a path to an stl or obj file to slice
        infill [str] - the geometric infill type (triangle, square, or hexagon)
        infill_size [float] - the density of the infill, roughly the distance between lattice in infill
        layer_height [float] - the height of each layer
        line_width [float] - the nozzle width of the 3D printer
    """
    def __init__(self, path: str, infill: str, infill_size: float, layer_height: float, line_width: float):
        self._model_path = path
        self._infill = infill
        self._infill_size = infill_size
        self._layer_height = layer_height
        self._line_width = line_width
    
    @property
    def path(self):
        return self._model_path

    @property
    def infill(self):
        return self._infill
    
    @property
    def infill_size(self):
        return self._infill_size

    @property
    def layer_height(self):
        return self._layer_height
    
    @property
    def line_width(self):
        return self._line_width


class Slicer:
    def __init__(self, params: Params) -> None:
        """
        Initialize the slicer using the parameters set
        """
        self.params = params


    def slice_debug(self) -> None:
        """
        Perform the exact same operations as the main slice function, but in order to save time in
        debugging and developing code, save major variables as pickle files and load them instead of
        re-running the whole code. 
        """

        # load the mesh and slice and perform a quick generation of the internal lattice 
        mesh = self.load_part(self.params.path)
        lattice = self.generate_lattice(mesh)
        layer_edges = self.create_raw_slices(mesh)
        
        self.n_layers = len(layer_edges)

        # base directory for all pickled variables
        base_dir = "pickled-vars/"

        # extra set of variables to force any section to re-run, even if pickle files exist
        force_slice_to_poly = False
        force_generate_layer_graphs = True
        force_generate_layer_paths = False

        # check to see if layer polygons have already been created
        layer_poly_file = base_dir + "layer_polygons.pckl"
        if os.path.isfile(layer_poly_file) and not force_slice_to_poly:
            with open(layer_poly_file, 'rb') as f:
                layer_polygons = pickle.load(f)
        else:
            layer_polygons = self.slice_to_polly(layer_edges)
            with open(layer_poly_file, 'wb') as f:
                pickle.dump(layer_polygons, f)

        # check to see if layer graphs have already been created
        layer_graphs_file = base_dir + "layer_graphs.pckl" 
        if os.path.isfile(layer_graphs_file) and not force_generate_layer_graphs:
            with open(layer_graphs_file, 'rb') as f:
                layer_graphs = pickle.load(f)
        else:
            layer_graphs = self.generate_layer_graphs(layer_polygons, lattice)
            with open(layer_graphs_file, 'wb') as f:
                pickle.dump(layer_graphs, f)
        
    def slice(self, debug_mode = False) -> None:

        if debug_mode:
            self.slice_debug()
            return

        mesh = self.load_part(self.params.path)
        lattice = self.generate_lattice(mesh)
        layer_edges = self.create_raw_slices(mesh)

        self.n_layers = len(layer_edges)

        # convert each slice of the 3D model into a convenient polygon object
        layer_polygons = self.slice_to_polly(layer_edges)

        # using the lattice and layer polygon, generate a graph for each layer
        self.layer_graphs = self.generate_layer_graphs(layer_polygons, lattice)

        # load in our planner object to do the actual path planning on the graph
        # self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        # generate the layer paths
        # self.layer_paths = self.planner.generate_layer_paths()            

        # generate a gcode file
        # self.planner.generate_gcode(self.layer_paths)

        # plot the finals paths
        # self.plot_final_paths()

    def load_part(self, path: str) -> None:
        """
        Preform all the necessary initializations when loading a model from a file
        """
        mesh = trimesh.load(path)
        mesh.rezero()

        return mesh

    def generate_lattice(self, mesh) -> Lattice:
        """
        Generate a set lattice for the infill
        """

        size = self.params.infill_size
        infill_type = self.params.infill

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

        # get bounds for mesh
        x_range = mesh.bounds[:,0]
        y_range = mesh.bounds[:,1]

        # build a lattice that is slightly larger than x_range and y_range of the model.
        latt.build((x_range[1] + 2*size, y_range[1] + 2*size), 
                   pos = (-size, -size))

        return latt
        
    def create_raw_slices(self, mesh) -> np.ndarray:
        """
        Take model and parameters and slice model uniformly along the xy axis. 
        
        Returns:
            np.ndarray
        """

        # get z bounds
        z_range = mesh.bounds[:,2]

        # create a vector that contains the height of each layer
        layer_heights = np.arange(0, z_range[1], self.params.layer_height)
        self.n_layers = np.size(layer_heights)

        # create plane origin and normal vector to plane
        origin = np.zeros((3))
        normal = np.array([0,0,1])

        # create slices starting at the origin, in the direction of the normal with
        # layer height specified
        layer_edges, _, _ = trimesh.intersections.mesh_multiplane(mesh, origin, 
                                                                  normal, layer_heights)
        
        return layer_edges

    def slice_to_polly(self, layer_edges) -> List[List[Polygon]]:
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
        for layer_edge in layer_edges:
            
            # reorder all the edge data, get back a list of distinct regions
            ordered_regions = self.reorder_edges(layer_edge)
            
            # convert each region in each layer to its own polygon object
            original_polygons = []
            for ring in ordered_regions:
                original_polygons.append(Polygon(ring,  edgecolor='b', facecolor='none'))

            scaled_polygons = []
            for polygon in original_polygons:
                poly_points = polygon.get_xy()

                buffer_poly = shp.Polygon(poly_points).buffer(-self.params.line_width)

                # annoyingly, this buffer polygon may accidentally split into more than one polygon
                # this can happen if there is a thin point (like a figure 8). When that happens, we
                # have to treat it as a different region
                if type(buffer_poly) == shp.MultiPolygon:
                    for sub_buffer_poly in buffer_poly.geoms:
                        scaled_polygons.append(Polygon(np.array(sub_buffer_poly.exterior.coords.xy).T))
                else:
                    scaled_polygons.append(Polygon(np.array(buffer_poly.exterior.coords.xy).T))

            slices.append(scaled_polygons)
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

    def generate_layer_graphs(self, layer_polygons, lattice) -> List[List[nx.Graph]]:
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

        for polygons in layer_polygons:
            graphs = []
            for polygon in polygons:
                # create graph of polygon
                G_polygon = nx.Graph()
                vertices = polygon.get_xy()
                
                graph_vertices = [(idx,{"x":val[0], "y":val[1]}) for idx,val in enumerate(vertices[:-1])]
                G_polygon.add_nodes_from(graph_vertices)

                graph_edges = [(i,i+1) for i in range(len(vertices) - 2)]
                graph_edges.append((0,len(graph_edges)))
                G_polygon.add_edges_from(graph_edges)

                nx.draw(G_polygon, pos=posgen(G_polygon), node_size = 1, with_labels=False)


                graphs.append([])
            layer_graphs.append(graphs)
            
            pbar.update(1)
            return layer_graphs

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
