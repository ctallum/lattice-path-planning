from slicer import Slicer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "./models/bunny.stl"

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "hexagon", # can also do "triangle" and "square"
        "infill_size": 5,
        "line_width": 0.2
    }

    slicer = Slicer(params)

    slicer.slice(path)

    plt.show()