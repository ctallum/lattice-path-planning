"""
File to contain Slicer class which takes a stl model and generates cross sections with different infill geometry
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Dict, List
import openmesh as om
import trimesh 
# from shapely import LinearRing, Point
from matplotlib.patches import Polygon



class Slicer:
    def __init__(self):
        """
        Initialize Slicer object
        """
        self.params_set = False

    def set_params(self, params: Dict) -> None:
        """
        Set common 3D printing parameters. Takes input dictionary with following values:
        Dict{
            "layer_height": float,
            "base_layers": int,
            "top_layers": int,
            "infill": str [cubic, hexagonal]
        }
        """
        self.layer_height = params["layer_height"]
        self.base_layers = params["base_layers"]
        self.top_layers = params["top_layers"]
        self.infill = params["infill"]
        
        self.params_set = True

    def slice(self, path: str) -> None:
        """
        Take input model path and slice according to pre-set parameters
        """
        self.path = path
        self.load_part()

        # Ensure that all parameters have been set before moving on
        if not self.params_set:
            print("ERROR: Please set slicer prameters")
            return
        
        self.create_slices()


        
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
        
    def create_slices(self) -> None:
        """
        Take model and parameters and slice model uniformly along the xy axis. 
        Creates self.layers_edges: List[np.ndarray(n_layers,2,2)]
        """

        layer_heights = np.arange(0, self.z_range[1], self.layer_height)
        self.n_layers = np.size(layer_heights)
        self.layer_edges, _, _ = trimesh.intersections.mesh_multiplane(self.mesh, np.zeros((3)), np.array([0,0,1]), layer_heights)

        self.edge_polygons = []
        for layer in range(self.n_layers):
            re_ordered_edge = self.reorder_edges(self.layer_edges[layer])
            self.all_edge_polygons.append(self.layer_to_polygon(re_ordered_edge))


    def reorder_edges(self,coordinates) -> List[np.ndarray]:
        """
        Iterate through all set of edges and re-order them. Also split into seperate rings if needed
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

    def layer_to_polygon(self, layer_edge: List[np.ndarray]) -> List[Polygon]:
        polygons = []
        for ring in layer_edge:
            polygons.append(Polygon(ring))

        return polygons

        
    def plot_layer_edge(self, layer: int) -> None:
        """
        Plot a given layer_edge
        Input: 
            layer: int
        """
        plt.figure()

        layer_edge = self.layer_edges[layer]

        for idx in range(np.shape(layer_edge)[0]):
            plt.plot(*layer_edge[idx,:,:].T, "-k")

    def calc_n_regions_layer(self, layer) -> int:
        """
        Calculate the number of distinct closed regions for any given layer
        """
        return len(self.edge_polygons[layer])

    def is_in_layer(self, layer: int, point: Tuple[float,float]):
        """
        Calculate if a given point is within the model at a given layer
        """
        return self.edge_polygons[layer].get_path().contains_point(point)

