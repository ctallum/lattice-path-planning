from slicer import Slicer
from matplotlib import pyplot as plt


if __name__ == "__main__":
    path = "./models/bunny.stl"

    slicer = Slicer()
    slicer.load(path)

    slicer.plot_mesh()


    plt.show()