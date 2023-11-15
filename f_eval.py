from stl import mesh


def f_eval(obj, a1, a2, par=None):

    # first make a local copy, prevent out of scope operations
    obj_rot = mesh.Mesh(obj.data.copy())

    # rotate obj sequentially by a1 and a2
    obj_rot.rotate([1, 0, 0], a1)
    obj_rot.rotate([0, 1, 0], a2)

    # perform function evaluation
    _, cog, _ = obj_rot.get_mass_properties(False)
    f = cog[-1]  # only interested in Z-direction
    return obj_rot, f
