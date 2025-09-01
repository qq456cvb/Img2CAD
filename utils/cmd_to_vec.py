import numpy as np
from math import pi, sqrt
import sys
sys.path.append('..')

from cadlib.macro import *
from cadlib.math_utils import polar_parameterization

GPT_DEEPCAD_ARG_CONV = {
    '<EOS>': 'EOS',
    '<SOL>': 'SOL',
    'L': 'Line',
    'A': 'Arc',
    'R': 'Circle',
    'E': 'Ext'
}

GPT_DEEPCAD_EXT_CONV = {
    'NewBody': 'NewBodyFeatureOperation',
    'Join': 'JoinFeatureOperation',
    'Intersect': 'IntersectFeatureOperation',
    'Cut': 'CutFeatureOperation',
    'OneSided': 'OneSideFeatureExtentType',
    'TwoSided': 'TwoSidesFeatureExtentType',
    'Symmetric': 'SymmetricFeatureExtentType'  # Not yet supported
}

''' IMPORTANT: READ FOR VECTOR FORMAT '''
# Each command becomes a size 17 array. -1 indicates null parameter.
#   Line: (2 params)
#       [0, x, y, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#   Arc: (4 params)
#       [1, x, y, alpha, f, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#   Circle: (3 params)
#       [2, x, y, -1, -1, r, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#   EOS: (0 params)
#       [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#   SOL: (0 params)
#       [4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
#   Ext: (11 params)
#       [5, -1, -1, -1, -1, -1, theta, gamma, lambda, px, py, pz, s, e1, e2, b, u]

''' IMPORTANT: READ FOR GPT COMMAND -> VECTOR CONVERSION '''
# The EOS and SOL vectors have no parameters, so they are easy to create from GPT.
# It is straightforward to convert line, arc, circle from the GPT format to vectors (same parameters).
# The only detail is that alpha in the vector command should be in radians, but GPT gives degrees.
# 
# The extrusion command conversion is more complex. Here are the key differences.
#   1) The first three arguments in GPT are (n1, n2, n3), the normal vector.
#      For vector, these are parametrized into (theta, gamma, lambda), 3D rotation matrix angles (radians).
#      THIS PARAMETRIZATION IS NOT UNIQUE, since GPT does not specify an x-axis in its sketch plane.
#      So we choose the x-axis in a fixed manner based on the normal axis, and use it to get the above angles.
#   2) GPT DOES NOT HAVE A SCALE ARGUMENT, BUT THE VECTOR FORMAT DOES (IN S).
#      We set this to a fixed value (95), based on experiments with OCC's renders.
#   3) We need to encode GPT's b and u parameters (extrusion type and operation).
#      These must be integers. We can use the two dictionaries at the top.

def get_x_from_normal(normal):
    n1, n2, n3 = normal
    x_axis_3d = np.array([-n2, n1, 0])
    if abs(n1) <= 1e-4 and abs(n2) <= 1e-4:
        x_axis_3d = np.array([-n3, 0, n1])
    x_axis_3d = x_axis_3d / (np.linalg.norm(x_axis_3d) + 1e-7)
    return x_axis_3d


# Given command type and arguments, converts it to vector.
# The flag `f` in (0,1) indicates for ARCS whether it should be clockwise or counterclockwise.
def cmd2vec(cmd_type, cmd_args, f=None, use_normal=True):
    # Invalid command type.
    if cmd_type not in GPT_DEEPCAD_ARG_CONV.keys():
        raise Exception("Invalid command type \'{}\'".format(cmd_type))

    # Convert start or end commands.
    if cmd_type == '<EOS>':
        return EOS_VEC.astype(float)
    if cmd_type == '<SOL>':
        return SOL_VEC.astype(float)

    # Initialize empty vector with relevant type in 0th entry.
    vec = np.full((N_ARGS + 1), PAD_VAL, dtype=float)
    cmd_idx = ALL_COMMANDS.index(GPT_DEEPCAD_ARG_CONV[cmd_type])
    vec[0] = cmd_idx
    args_idx = 1 + np.nonzero(CMD_ARGS_MASK[cmd_idx])[0]
    
    # Sketch commands.
    if cmd_type in ['L', 'A', 'R', 'C']:
        # try:
        vec[args_idx] = np.array(cmd_args, dtype=float)

        # For arcs, need to make degrees into radians.
        if cmd_type == 'A':
            if f != None:
                vec[4] = f
        return vec

        # except Exception as e:
        #     print(f"Invalid number of arguments for {cmd_type}")
        #     return

    # Extrude command.
    # GPT format: (n1, n2, n3, nx1, nx2, nx3, x, y, z, _, e1, e2, b, u)
    # VEC format: (th, ph, ga, x, y, z, s, e1, e2, b, u)
    try:
    # print(cmd_args, len(cmd_args))
        offset = 0
        if use_normal:
            offset = 3  # new format need two directions
            normal_3d = np.array(cmd_args[:3], dtype=float)
            normal_3d = normal_3d / (np.linalg.norm(normal_3d) + 1e-7)
            x_axis_3d = np.array(cmd_args[3:6], dtype=float)
            
            # project x axis to orthogonal to normal
            x_axis_3d = x_axis_3d - np.dot(x_axis_3d, normal_3d) * normal_3d
            if np.linalg.norm(x_axis_3d) < 1e-4:
                x_axis_3d = get_x_from_normal(normal_3d)
            x_axis_3d = x_axis_3d / (np.linalg.norm(x_axis_3d) + 1e-7)

            # Normalization
            theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
            vec[args_idx[:3]] = np.array([theta, phi, gamma], dtype=float)
        else:
            vec[args_idx[:3]] = np.array(cmd_args[:3], dtype=float)

        # Next three arguments (x, y, z) are the same
        vec[args_idx[3:6]] = np.array(cmd_args[3 + offset:6 + offset], dtype=float)

        # Set scale (s)
        vec[args_idx[6]] = 1.

        # Next two arguments (e1, e2) are the same
        vec[args_idx[7:9]] = np.array(cmd_args[6 + offset:8 + offset], dtype=float)

        # Next two arguments (b, u) are modified to match DeepCAD format
        b, u = cmd_args[8 + offset:10 + offset]
        vec[args_idx[9]] = EXTRUDE_OPERATIONS.index(GPT_DEEPCAD_EXT_CONV[b])
        vec[args_idx[10]] = EXTENT_TYPE.index(GPT_DEEPCAD_EXT_CONV[u])

    except Exception as e:
        print("Extrusion error: ", e)
        return

    return vec

''' MAKING CHANGES AT THE VECTOR LEVEL '''

# Checks arc angles are reasonable because these can explode bbox measurements.
def arc_angle_fix(cad_vec):
    new_cad_vec = []
    for cmd_vec in cad_vec:
        cmd_type = ALL_COMMANDS[int(cmd_vec[0])]
        
        # TODO: Fix this handling later, for now just set to pi/2.
        if cmd_type == 'Arc':
            cmd_vec[3] = pi / 2

        new_cad_vec.append(cmd_vec)
    return np.array(new_cad_vec)



# Alter cad program to match SCALE, given its aabbox and desired bbox.
def match_cad_scale(cad_vec, aabbox, bbox):
    eps = 1e-2
    scalex = bbox[3] / max(aabbox[3], eps)
    scaley = bbox[4] / max(aabbox[4], eps)
    scalez = bbox[5] / max(aabbox[5], eps)
    new_cad_vec = []

    for cmd_vec in cad_vec:
        cmd_type = ALL_COMMANDS[int(cmd_vec[0])]

        # For lines and arcs, need to scale the x and y coords according to scalex, scaley.
        if cmd_type in ['Line', 'Arc']:
            cmd_vec[1] *= scalex
            cmd_vec[2] *= scaley

        # For circle, need to scale radius by geometric mean of scalex and scaley.
        elif cmd_type in ['Circle']:
            cmd_vec[5] *= sqrt(abs(scalex * scaley))

        # For extrusions, need to scale extrusion distances.
        elif cmd_type in ['Ext']:
            cmd_vec[13:15] *= scalez

        new_cad_vec.append(cmd_vec)

    return np.array(new_cad_vec)

# Alter cad program to match ROTATION.
def match_cad_rot(cad_vec, aabbox, bbox):
    x_axis = bbox[6:9] / np.linalg.norm(bbox[6:9])
    y_axis = bbox[9:] / np.linalg.norm(bbox[9:])
    z_axis = np.cross(x_axis, y_axis)
    transform = np.array([x_axis, y_axis, z_axis])
    pargs = polar_parameterization(z_axis, x_axis)

    new_cad_vec = []

    for cmd_vec in cad_vec:
        cmd_type = ALL_COMMANDS[int(cmd_vec[0])]

        # Only change the Extrusion coordinate system.
        if cmd_type in ['Ext']:
            cmd_vec[6:9] = np.array(pargs, dtype=float)

        new_cad_vec.append(cmd_vec)

    return np.array(new_cad_vec)

# Alter cad program to match POSITION.
def match_cad_pos(cad_vec, aabbox, bbox):
    center_diff = bbox[:3] - aabbox[:3]

    new_cad_vec = []

    for cmd_vec in cad_vec:
        cmd_type = ALL_COMMANDS[int(cmd_vec[0])]

        # Only change the extrusion origin.
        if cmd_type in ['Ext']:
            cmd_vec[9:12] += center_diff

        new_cad_vec.append(cmd_vec)

    return np.array(new_cad_vec)