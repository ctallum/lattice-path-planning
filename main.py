from slicer import Slicer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "./models/bunny.stl"

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "triangle",
        "infill_size": 10
    }

    slicer = Slicer(params)

    slicer.slice(path)

    plt.show()