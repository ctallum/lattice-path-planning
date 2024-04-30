from slicer import Slicer
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pickle


if __name__ == "__main__":

    demo_files = {"211": "bunny_large_hex", 
                  "201": "bunny_small_hex",
                  "011": "bunny_large_triangle",
                  "001": "bunny_small_triangle",
                  "111": "bunny_large_square",
                  "101": "bunny_small_square",
                  "210": "cube_large_hex",
                  "200": "cube_small_hex",
                  "010": "cube_large_triangle",
                  "000": "cube_small_triangle",
                  "110": "cube_small_square",
                  "100": "cube_large_square",
                  }


    state = {"infill": "0", "size": "0", "model": "0", "range": 100}

    demo_data = {}

    for key, value in demo_files.items():
        f = open(f'demo-files/{value}.pckl', 'rb')
        layer_paths = pickle.load(f)
        f.close()
        demo_data[key] = layer_paths


    def plot_model():
        ax.clear()
        key = state["infill"] + state["size"] + state["model"]
        layer_paths = demo_data[key] 

        n_layers = len(layer_paths)
        upper_val = int((state["range"] * n_layers) / 100 )

        for layer_idx, layer_data in enumerate(layer_paths[0:upper_val]):
            for pts in layer_data:
                ax.plot(*pts.T, layer_idx*0.2)
        
        plt.axis("equal")
        fig.canvas.draw_idle()


    def plot_hex(event):
        if state["infill"] != "2":
            state["infill"] = "2"
            plot_model()

    def plot_tri(event):
        if state["infill"] != "0":
            state["infill"] = "0"
            plot_model()

    def plot_sq(event):
        if state["infill"] != "1":
            state["infill"] = "1"
            plot_model()

    def plot_small(event):
        if state["size"] != "0":
            state["size"] = "0"
            plot_model()

    def plot_large(event):
        if state["size"] != "1":
            state["size"] = "1"
            plot_model()

    def plot_cube(event):
        if state["model"] != "0":
            state["model"] = "0"
            plot_model()

    def plot_bunny(event):
        if state["model"] != "1":
            state["model"] = "1"
            plot_model()


    # Create initial plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    

    plot_model()

    ax_hex = plt.axes([0.82, 0.05, 0.1, 0.075])
    ax_tri = plt.axes([0.6, 0.05, 0.1, 0.075])
    ax_sq = plt.axes([0.71, 0.05, 0.1, 0.075])
    hex_button = Button(ax_hex, 'Hexagon')
    tri_button = Button(ax_tri, 'Triangle')
    sq_button = Button(ax_sq, "Square")

    hex_button.on_clicked(plot_hex)
    tri_button.on_clicked(plot_tri)
    sq_button.on_clicked(plot_sq)

    ax_small = plt.axes([0.35, 0.05, 0.1, 0.075])
    ax_large = plt.axes([0.46, 0.05, 0.1, 0.075])

    small_button = Button(ax_small, "Small")
    large_button = Button(ax_large, "Large")

    small_button.on_clicked(plot_small)
    large_button.on_clicked(plot_large)

    ax_cube = plt.axes([0.1, 0.05, 0.1, 0.075])
    ax_bunny = plt.axes([0.21, 0.05, 0.1, 0.075])

    cube_button = Button(ax_cube, "Cube")
    bunny_button = Button(ax_bunny, "Bunny")

    cube_button.on_clicked(plot_cube)
    bunny_button.on_clicked(plot_bunny)

    # Initial range
    range_min = 0
    range_max = 100
    initial_lower_value = range_min
    initial_upper_value = range_max

    # Create sliders for lower and upper bounds
    slider_upper_ax = plt.axes([0.1, 0.25, 0.02, 0.65], facecolor='lightgray')

    slider_upper = Slider(slider_upper_ax, 'Percent of Layers Shown', range_min, range_max, valinit=initial_upper_value, orientation='vertical')

    # Define update function for the sliders
    def update_slider(val):
        # get n_layers
        state["range"] =  slider_upper.val

        plot_model()

    slider_upper.on_changed(update_slider)



    plt.show()