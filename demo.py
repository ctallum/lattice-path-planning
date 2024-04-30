from slicer import Slicer
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np
import pickle


if __name__ == "__main__":

    demo_files = {
        "211": "bunny_large_hex", "201": "bunny_small_hex",
        "011": "bunny_large_triangle", "001": "bunny_small_triangle",
        "111": "bunny_large_square", "101": "bunny_small_square",
        "210": "cube_large_hex", "200": "cube_small_hex",
        "010": "cube_large_triangle", "000": "cube_small_triangle",
        "110": "cube_small_square", "100": "cube_large_square"
    }

    state = {"infill": "0", "size": "0", "model": "0", "range": 100}

    demo_data = {}

    for key, value in demo_files.items():
        with open(f'demo-files/{value}.pckl', 'rb') as f:
            layer_paths = pickle.load(f)
        demo_data[key] = layer_paths

    def plot_model():
        ax.clear()
        key = state["infill"] + state["size"] + state["model"]
        layer_paths = demo_data[key] 

        n_layers = len(layer_paths)
        upper_val = int((state["range"] * n_layers) / 100 )

        upper_val = max(1,upper_val)

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

    def update_range(val):
        try:
            val = int(val)
            if 0 <= val <= 100:
                state["range"] = val
                slider.set_val(val)
                plot_model()
            else:
                print("Value must be between 0 and 100.")
        except ValueError:
            print("Please enter a valid integer.")

    def update_slider(val):
        range_text_box.set_val(str(int(val)))
        state["range"] = int(val)
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

    # Create slider for range
    slider_ax = plt.axes([0.1, 0.25, 0.02, 0.65], facecolor='lightgray')
    slider = Slider(slider_ax, 'Range', 0, 100, valinit=state["range"], valstep=5,orientation="vertical")
    slider.on_changed(update_slider)

    # Create text input for range
    range_text_box_ax = plt.axes([0.09, 0.18, 0.05, 0.03])
    range_text_box = TextBox(range_text_box_ax, '', initial=str(state["range"]))
    range_text_box.on_submit(update_range)

    plt.show()
