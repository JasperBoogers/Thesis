# Gradient based optimization of part orientation in 3D printing
![](/assets/armadillo_opt_steps.gif)

## Background
For my Master's thesis in Mechanical Engineering, I investigated how gradient based optimization can be used to find the optimal part orientation in a 3D printing environment.
For simplicity, I only considered the volume of the support structures as the objective, although extensions are relatively easy, as long as you can calculate derivative information. 
The resultant thesis can be retrieved from the [TU Delft Repository](https://repository.tudelft.nl/record/uuid:238df854-ec9f-4cd3-a7b6-e05adb86be54)

## Specific package requirements
[PyVista](https://docs.pyvista.org) is used for creating, cleaning, and rotating mesh objects. PyVista is a wrapper of many [VTK](https://github.com/Kitware/VTK) functions, which is the standard in geometry processing.
PyVista makes using VTK much more intuitive and comprehensive to use, in my opinion.

Currently (July 2024), PyVista does have some limitations. For instance, each point coordinate in the mesh is only stored as a single precision number. This results in loss of precision when rotating and translating, even for small objects. See [this discussion](https://github.com/pyvista/pyvista/discussions/4128) for an example. As an intermediate solution, I downscaled all the mesh objects and loosened the tolerance on the gradient during optimization.

For other required packages, see the [requirements file](\requirements.txt)

## Repository contents
This repo contains all code necessary to replicate the work outlined in my thesis. Each of the following subsections discuss one chapter in the thesis.
Frequently used methods by multiple sections are all extracted to the `helpers` directory

### Convex shapes
Generates all figures for chapter 3 of the thesis. Only convex shapes are considered, where on-part support cannot occur.  

### Non-convex shapes

### Case study

### Time estimation
For this code to run, **SuperSlicer is required**. Download the [latest release](https://github.com/Kitware/VTK) and place the entire 'SuperSlicer' folder in the time_estimation directory.
