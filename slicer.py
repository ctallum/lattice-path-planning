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
import shapely as shp
from collections import defaultdict
import pickle
import os
import math

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

def posgen(G):
    ret = {}
    for n in G:
        ret[n] = [G.nodes[n]["x"],G.nodes[n]["y"]]
    return ret
