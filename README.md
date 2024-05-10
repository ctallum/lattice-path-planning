# Lattice-Path-Planning
Path planning for continious 3D printing of lattice patterns using planar graph tracing

### Summary of Files
main.py - the main file if you want to slice an object and generate a path/gcode

slicer.py - a helper file that does a lot of the slicing operations

planner.py - a helper file that does a lot of the path planning operations

### Summary of Demo files
demo.py - A demo program that shows you different pre-sliced models with different infills

enumdemo.py - A demo program that enumerates over every spanning tree for a simple undirected graph

### Misc Files
demo files directory contains picked python variables that are used in demo.py

cython directory contains another main.py file that does the exact same thing as the original, but it relies on some cython magic, so it tends to be quite a bit quicker to run
