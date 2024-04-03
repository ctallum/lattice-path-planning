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
from lattpy import Shape



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
        self.path = path
        self.load_part()
        self.layer_edges = self.create_raw_slices()
        # self.slice_polygons = self.slice_to_polly(self.layer_edges)

        self.lattice = self.generate_lattice()
        self.plot_layer_edge(200)

        
    def load_part(self) -> None:
        """
        Preform all the necessary initializations when loading a model from a file
        """
        self.mesh = trimesh.load(self.path)
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

    def generate_lattice(self) -> Shape:
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

        ax = latt.plot()
        s.plot(ax)

        return s







