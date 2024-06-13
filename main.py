from slicer import Slicer, Params
from matplotlib import pyplot as plt



if __name__ == "__main__":
    path = "./models/bunny.stl"

    params = Params(
        path = path,
        infill = "triangle", # can also do "triangle" and "square"
        infill_size = 5.0,
        layer_height = 0.2,
        line_width= 0.4
    )

    slicer = Slicer(params)

    slicer.slice(True)

    plt.show()