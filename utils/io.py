# Writes stl to obj file.
from OCC.Extend.DataExchange import write_stl_file
import numpy as np
import trimesh
import os


def stl_to_obj(stl_file, obj_file):
    trimesh.load_mesh(stl_file).export(obj_file)

# Converts CAD solid to obj file through intermediate stl.
def cad_to_obj(shape, obj_file):
    stl_file = obj_file[:-3] + 'stl'
    write_stl_file(shape, stl_file)
    stl_to_obj(stl_file, obj_file)
    os.system("rm " + stl_file)