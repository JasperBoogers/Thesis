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

For other required packages, see the [requirements file](requirements.txt)

## Repository contents
This repo contains all code necessary to replicate the work outlined in my thesis. Each of the following subsections discuss one chapter in the thesis.
Frequently used methods by multiple sections are all extracted to the `helpers` directory

### Convex shapes
Generates all figures for chapter 3 of the thesis. Only convex shapes are considered, where on-part support cannot occur. The convex shapes are split in 2D and 3d shapes.
At first, I thought of using Pyvista to extrude the volume below each overhanging facet and summing those volumes. However, that generates many small volumes, which might (not) intersect slightly. If you then try to combine each of the volume elements, the resultant support volume is bogus.

### Non-convex shapes
Generates all figures for chapter 4 of the thesis, where now on-part support is considered and handled as well. First, a mask is generated based on wether each facet requires support or not. The mask is then multiplied with the volume below _each_ facet and summed to return the total support volume.

### Case study
Generates all figures for chapter 5, which are some numerical experiments performed using the code for non-convex shapes. Experiments include parameter studies, effects of mesh size, and a performance comparison between gradient based optimizers and population based algorithms.

### Time estimation
Something I did not have time for, but what could be an interesting extension: rather than minimizing the support volume (or whatever function associated with cost), minimize the time required to produce a certain part.
For that to work, you would need a time estimator, as running each orientation through a slicer would take forever. I briefly investigated this problem by setting up a time estimator that uses polynomial regression to come up with a relation between the part volume, height, and area and the slicer time. The estimator worked fairly well, even with a very low number of input. In the end, working out how to compute the support volume of non-convex shapes in a smooth way (otherwise gradient based optimization returns useless results) took much longer than expected, and I did not get around this bit.

N.B. For this code to run, **SuperSlicer is required**. Download the [latest release](https://github.com/Kitware/VTK) and place the entire 'SuperSlicer' folder in the time_estimation directory.
