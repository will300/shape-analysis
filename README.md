# shape-analysis

This repository houses all files making up part of my masters project, which is focused on the analysis of CAD models. The aim of the project is to create an automatic system for segmenting and classifying different components within models, e.g. shafts and holes, using convolutional neural networks. The current stage of the project is conversion of triangular mesh style CAD files into square grids which are much more suitable as input for neural networks 

## dataset

ModelNet10, which can be downloaded [here](http://modelnet.cs.princeton.edu/), is currently being used for testing. It has thousands of training models belonging to ten distinct categories.   

## modules

Tensorflow 1.12.0; Numpy 1.16.0; Matplotlib 3.0.2, Keras 2.2.0; Python 3.6

## results

Running main.py in its current state should yield the following plots, which show different processing of the same CAD model. Both 2D and 3D representations are included, along with an image of the original mesh. The processing resolution and direction of view can be changed in some of the functions. 



Other files have been included which can also be inputted, however some processing times are many minutes. The next phase of the project will be to convert the functions to C++ for greater speeds suitable for processing a large dataset.
