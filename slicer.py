"""
File to contain Slicer class which takes a stl model and generates cross sections with different infill geometry
"""

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
import openmesh as om
import trimesh 

class Slicer:
    def load(self, stl_path: str) -> None:
        """
        Preform all the necessary initializations when loading a model from a file
        """
        self.path = stl_path
        self.mesh = trimesh.load(self.path)
        self.mesh.rezero()
        self.bound = self.mesh.bounds
            

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

    def create_slices(self, slice_height: float = 0.2) -> None:
        pass