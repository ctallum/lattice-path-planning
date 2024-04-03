from slicer import Slicer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "./models/bunny.stl"

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "cubic"
    }

    slicer = Slicer(params)

    slicer.slice(path)

    # print(slicer.n_layers)

    # slicer.plot_layer_edge(530)


    plt.show()