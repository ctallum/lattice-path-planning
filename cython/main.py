from classes import *
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "../models/20mm_cube.stl"

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "square", # can also do "triangle" and "square"
        "infill_size": 4,
        "line_width": 0.4
    }

    slicer = Slicer(params)

    slicer.slice(path)

    plt.show()