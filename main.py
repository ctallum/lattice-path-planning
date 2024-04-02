from slicer import Slicer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "./models/bunny.stl"

    slicer = Slicer()

    params = {
        "layer_height": 0.2,
        "base_layers": 2,
        "top_layers": 2,
        "infill": "cubic"
    }
    
    slicer.set_params(params)
    
    slicer.slice(path)


    plt.show()