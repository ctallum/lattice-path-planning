"""
File to contain Slicer class which takes a stl model and generates cross sections with different infill geometry
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
from collections import defaultdict
import osmnx

from planner import Planner

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

        self.n_layers = 1 # for debug, to keep runtime down

        # layer cleanup to create graphs
        self.layer_polygons = self.slice_to_polly(self.layer_edges)
        self.layer_graphs = self.generate_layer_graphs(self.lattice, self.layer_polygons)


        # self.plot_layer_graph(0)

        self.planner = Planner(self.params, self.layer_polygons, self.layer_graphs)

        self.planner.plan()

        
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
                        alpha=.4)
        
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
            re_ordered_edge = self.reorder_edges(layer_edges[layer])
            polygons = []
            for ring in re_ordered_edge:
                polygons.append(Polygon(ring))
            slices.append(polygons)
            pbar.update(1)

        return slices


    def reorder_edges(self,coordinates) -> List[np.ndarray]:
        """
        Iterate through all set of edges and re-order them. Also split into separate rings if needed
        """
        coordinates = np.around(coordinates,4).tolist()
        reordered_coords = []
        while coordinates:
            current_edge = coordinates[0]
            coordinates.pop(0)
            reordered_region = [current_edge]
            while True:
                next_edge_index = None
                for i, edge in enumerate(coordinates):
                    if edge[0] == current_edge[1]:
                        next_edge_index = i
                        reordered_region.append(edge)
                        current_edge = edge
                        coordinates.pop(i)
                        break
                    if edge[1] == current_edge[1]:
                        next_edge_index = i
                        reordered_region.append(edge[::-1])
                        current_edge = edge[::-1]
                        coordinates.pop(i)
                        break
                if next_edge_index is None:
                    break
            reordered_coords.append(reordered_region)
            reordered_coords = [np.array(a) for a in reordered_coords]
            edge_rings = []
            for ring in reordered_coords:
                edge_rings.append(ring[:,0,:])

        return edge_rings
        
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

        # ax = latt.plot()
        # s.plot(ax)

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

                # get all bounding polygon points
                poly_points = polygon.get_xy()
                
                # iterate through all polygon edges and lattice edges to find intersecting sets
                problem_edges = defaultdict(list)

                # calc ahead of time all 

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
                    valid_mids =  polygon.get_path().contains_points(np.array(mid_points))

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
                inside_points = polygon.get_path().contains_points(np.array(lat_points))

                bad_points_idx = [idx for idx,val in enumerate(inside_points) if not val and lat_points[idx] not in critical_points]
                
                # only add edges to final edges if they do not touch outside points
                for edge in cleaned_edges:
                    if edge[0] not in bad_points_idx and edge[1] not in bad_points_idx:
                        final_edges.append(edge)


                # add valid lattice points to final points
                offset = len(lat_points)
                final_points = lat_points 
                # final_edges = lat_edges

                # for idx,edge in enumerate(new_edges):
                #     new_edges[idx][0] += offset
                #     new_edges[idx][1] += offset

                # # final_edges = new_edges


                # add all polygon edges to final edges and final points
                # offset = len(final_points)

                # for poly_point_idx in range(len(poly_points) - 1):
                #     A = poly_points[poly_point_idx ,:]
                #     B = poly_points[poly_point_idx + 1,:]
                #     final_points.append(A)
                #     final_points.append(B)
                #     final_edges.append([offset + poly_point_idx*2, offset + poly_point_idx*2 + 1])

                # create graph
                G = nx.Graph()

                # create networkx vertex and edge data types
                graph_vertices = [(idx,{"x":val[0], "y":val[1]}) for idx,val in enumerate(final_points)]
                graph_edges = [tuple(i) for i in final_edges]

                G.add_nodes_from(graph_vertices)
                G.add_edges_from(graph_edges)

                # remove all isolated vertices
                G.remove_nodes_from(list(nx.isolates(G)))

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
            
        


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def posgen(G):
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
