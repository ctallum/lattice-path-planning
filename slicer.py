"""
File to contain Slicer class which takes a stl model and generates cross sections with different infill geometry
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Dict
import openmesh as om
import trimesh 


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


