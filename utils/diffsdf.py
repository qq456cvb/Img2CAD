# %%
import trimesh
import numpy as np
from pathlib import Path
import shutil
import subprocess
import os
import sys
import h5py
from cadlib.visualize import vec2CADsolid
from cadlib.macro import *
from cadlib.math_utils import polar_parameterization_inverse
import torch
from tqdm import tqdm
from utils.io import cad_to_obj
import cv2
from cadlib.visualize import vec2CADsolid
import math


def rads_to_degs(rads: torch.Tensor) -> torch.Tensor:
    """Convert angle from radians to degrees"""
    return 180.0 * rads / math.pi


def angle_from_vector_to_x(vec: torch.Tensor) -> torch.Tensor:
    """Compute angle (0~2pi) between a unit 2D vector and positive x-axis"""
    x, y = vec[0], vec[1]
    # atan2 returns (-pi, pi]
    ang = torch.atan2(y, x)
    return ang % (2 * math.pi)


def cartesian2polar(vec: torch.Tensor, with_radius: bool=False):
    """Convert 3D cartesian vector to spherical (theta, phi[, r])"""
    vec = vec.to(torch.float64)
    norm = torch.norm(vec)
    theta = torch.acos(vec[2] / (norm + 1e-7))
    phi = torch.atan2(vec[1], vec[0])  # (-pi, pi]
    if not with_radius:
        return torch.stack([theta, phi])
    return torch.stack([theta, phi, norm])


def polar2cartesian(sph: torch.Tensor) -> torch.Tensor:
    """Convert spherical (theta, phi[, r]) to 3D cartesian"""
    theta, phi = sph[0], sph[1]
    r = sph[2] if sph.numel() == 3 else 1.0
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z])


def rotate_by_x(vec: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(th), torch.sin(th)
    R = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=vec.dtype, device=vec.device)
    return R @ vec


def rotate_by_y(vec: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(th), torch.sin(th)
    R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=vec.dtype, device=vec.device)
    return R @ vec


def rotate_by_z(vec: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ph), torch.sin(ph)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=vec.dtype, device=vec.device)
    return R @ vec


def polar_parameterization_inverse_torch(theta: torch.Tensor,
                                         phi: torch.Tensor,
                                         gamma: torch.Tensor):
    """Build coordinate frame from spherical angles"""
    # normal is z-axis of new frame
    normal = polar2cartesian(torch.stack([theta, phi]))
    # reference x after rotating standard x by theta then phi

    one_vec = torch.tensor([1.0, 0.0, 0.0], device=theta.device, requires_grad=False)
    ref_x = rotate_by_z(rotate_by_y(one_vec, theta), phi)
    # y-axis is cross of normal and ref_x
    ref_y = torch.cross(normal, ref_x, dim=0)
    # x-axis after additional rotation gamma around normal
    x_axis = ref_x * torch.cos(gamma) + ref_y * torch.sin(gamma)
    return normal, x_axis


def roty(a):
    return np.array([[np.cos(a), 0, -np.sin(a), 0],
                        [0, 1, 0, 0],
                        [np.sin(a), 0, np.cos(a), 0],
                        [0, 0, 0, 1]])
    
def rotx(a):
    return np.array([[1, 0, 0, 0],
                        [0, np.cos(a), -np.sin(a), 0],
                        [0, np.sin(a), np.cos(a), 0],
                        [0, 0, 0, 1]])


# %%
def append_faces(vertices_seq, faces_seq):
    vertices = torch.cat(vertices_seq, dim=0)
    cnt = 0
    for i in range(len(faces_seq)):
        faces_seq[i] += cnt
        cnt += vertices_seq[i].shape[0]
        
    faces = torch.cat(faces_seq, 0)
    return vertices, faces
    
def extrude_convex_mesh(verts, faces, direction):
    # assert mesh.is_convex, "Can only extrude convex meshes"

    normal = direction / torch.norm(direction)

    # stack the (n,3) faces into (3*n, 2) edges
    edges = torch.cat([faces[:, [1, 0]],
                          faces[:, [2, 1]],
                          faces[:, [0, 2]]], dim=0)
    # print(edges.shape)
    # with torch.no_grad():
    edges_sorted, edges_sorted_idx = torch.sort(edges, dim=1)
    edges_unique, edges_inverse_idx, edges_counts = torch.unique(edges_sorted, dim=0, return_inverse=True, return_counts=True)
    # edges_inverse_idx = edges_inverse_idx.int()
    # edges_sorted_idx = edges_sorted_idx.int()
    # print(edges_unique.dtype, edges_inverse_idx.dtype, edges_sorted_idx.dtype)
    edges_unique_sorted_idx = torch.scatter(torch.zeros((edges_unique.shape[0], 2)).to(edges_unique).long(), 0, edges_inverse_idx[:, None].expand(-1, 2), edges_sorted_idx)
    # print(edges_unique_sorted_idx)
    edges_unique = torch.gather(edges_unique, 1, edges_unique_sorted_idx)
    edges_unique = edges_unique[edges_counts == 1]

    boundary = verts[edges_unique]  # n, 2, 3
    # we are creating two vertical triangles for every 3D line segment
    # on the boundary of the 3D triangulation
    # vertical = np.tile(boundary.reshape((-1, 3)), 2).reshape((-1, 3))
    # vertical = boundary.reshape(-1, 3).repeat(1, 2).reshape(-1, 3)
    # vertical[1::2] += direction
    # build the repeated vertices
    verts_rep = boundary.reshape(-1,3).repeat(1,2).reshape(-1,3)

    # mask for odd rows
    idx = torch.arange(verts_rep.shape[0], device=verts_rep.device)
    mask = (idx % 2 == 1).unsqueeze(-1)  # shape (N,1)

    # add the extrusion offset only on the odd rows, without in‑place
    vertical = verts_rep + mask.float() * direction.unsqueeze(0)
    # vertical_faces = np.tile([3, 1, 2, 2, 1, 0],
    #                          (len(boundary), 1))
    vertical_faces = torch.tensor([3, 1, 2, 2, 1, 0], dtype=torch.long).to(verts.device).repeat(len(boundary), 1)  # n, 6
    vertical_faces += torch.arange(boundary.shape[0]).reshape(-1, 1).to(vertical_faces) * 4
    vertical_faces = vertical_faces.reshape(-1, 3)

    # reversed order of vertices, i.e. to flip face normals
    top_faces_seq = torch.flip(faces, dims=(1,))
    bottom_faces_seq = faces.clone()
    # top_faces_seq = faces.clone()

    # a sequence of zero- indexed faces, which will then be appended
    # with offsets to create the final mesh
    faces_seq = [bottom_faces_seq,
                 top_faces_seq,
                 vertical_faces]
    # print(bottom_faces_seq.shape, vertical_faces.shape)
    # print(bottom_faces_seq.shape, vertical_faces.shape)
    vertices_seq = [verts,
                    verts.clone() + direction,
                    vertical]

    # with torch.no_grad():
        # append sequences into flat nicely indexed arrays
    vertices, faces = append_faces(vertices_seq, faces_seq)

    return vertices, faces

# %%
# size: N x 2, first control the scale of 2D sketch, second control the extrusion height
# loc: N x 3, control the extrusion position
def param2mesh(part_names, part_cad_vec, size, loc):
    verts_all, faces_all = [], []
    for part_idx, (part_name, part_vec) in enumerate(zip(part_names, part_cad_vec)):
        verts, faces = [], []
        ext_dir = None
        ext_x = None
        ext_y = None
        ext_origin = None
        ext_dist = None
        for vec in part_vec:
            cmd_type = int(vec[0] + 0.5)
            if ALL_COMMANDS[cmd_type] == 'Line':
                verts.append(np.array([float(v) for v in vec[1:3]]))
            elif ALL_COMMANDS[cmd_type] == 'Arc':
                # continue
                if len(verts) == 0:
                    raise ValueError("Arc command must be preceded by a line command.")
                
                end_point = np.array([float(v) for v in vec[1:3]])
                sweep_angle = float(vec[3]) / 180 * np.pi
                clock_sign = int(vec[4])
                start_point = verts[-1]
                s2e_vec = end_point - start_point
                if np.linalg.norm(s2e_vec) == 0:
                    raise ValueError("Start and end points are the same; cannot define an arc.")

                # Calculate radius and other parameters
                radius = (np.linalg.norm(s2e_vec) / 2) / (np.sin(sweep_angle / 2) + 1e-8)
                s2e_mid = (start_point + end_point) / 2
                vertical = np.cross(s2e_vec, [0, 0, 1])[:2]
                vertical = vertical / (np.linalg.norm(vertical) + 1e-8)
                if clock_sign == 0:
                    vertical = -vertical

                # Calculate the center of the arc
                center_point = s2e_mid - vertical * (radius * np.cos(sweep_angle / 2))

                # Determine the reference vector and angles
                if clock_sign == 0:
                    ref_vec = end_point - center_point
                else:
                    ref_vec = start_point - center_point

                ref_vec = ref_vec / np.linalg.norm(ref_vec)

                # Generate discrete points along the arc
                res = 20
                angles = np.linspace(sweep_angle / res, sweep_angle, res - 1, endpoint=True)  # Linearly spaced angles
                points = []

                for angle in angles:
                    rot_matrix = np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]
                    ])
                    arc_point = center_point + rot_matrix @ (ref_vec * radius)
                    points.append(arc_point)
                verts.extend(points)
            elif ALL_COMMANDS[cmd_type] == 'Circle':
                center = np.array([float(v) for v in vec[1:3]])
                radius = float(vec[5])
                res = 40
                angles = np.linspace(0, 2 * np.pi, res, endpoint=False)
                points = []
                for angle in angles:
                    arc_point = center + np.array([np.cos(angle), np.sin(angle)]) * radius
                    points.append(arc_point)
                verts.extend(points)
            else:
                cmd_args = vec[1:][np.where(CMD_ARGS_MASK[cmd_type])[0]]
                if ALL_COMMANDS[cmd_type] == 'Ext':
                    cmd_args = np.concatenate([cmd_args[:6], cmd_args[7:8]])
                    ext_dir, ext_x = polar_parameterization_inverse(cmd_args[0], cmd_args[1], cmd_args[2])
                    ext_y = np.cross(ext_dir, ext_x)
                    cmd_args[:3] = ext_dir
                    ext_origin = np.array(cmd_args[3:6])
                    ext_dist = cmd_args[-1]
                    # print(name_mapping[ALL_COMMANDS[cmd_type]] + ':', '(' + ', '.join(['{:.02f}'.format(a) for a in cmd_args]) + ', NewBody, OneSided)')
            
        # add faces to form a mesh
        for i in range(1, len(verts) - 1):
            faces.append([0, i + 1, i])
        faces = torch.from_numpy(np.array(faces)).int().to(size.device)
        verts = torch.from_numpy(np.stack(verts)).float().to(size.device) * size[part_idx, 0]
        verts = verts[:, 0:1] * torch.from_numpy(ext_x).to(verts) \
            + verts[:, 1:2] * torch.from_numpy(ext_y).to(verts) \
            + torch.from_numpy(ext_origin).to(verts) + loc[part_idx]

        ext_dir = torch.from_numpy(ext_dir).to(verts)
        # print(verts.shape, faces.shape, ext_dir, ext_dist)
        verts_ext, faces_ext = extrude_convex_mesh(verts, faces, torch.tensor(ext_dir * ext_dist * size[part_idx, 1], dtype=torch.float32))
        verts_all.append(verts_ext)
        faces_all.append(faces_ext)
    
    verts_all, faces_all = append_faces(verts_all, faces_all)
    return verts_all, faces_all.int()


def param2mesh_torch(part_names, part_cad_vec, size, loc, rad=False):
    device = size.device
    eps = 1e-8
    verts_all = []
    faces_all = []

    for part_idx, part_name in enumerate(part_names):
        part_vec = part_cad_vec[part_idx]         # Tensor of shape [num_cmds, vec_len]
        verts = []
        faces = []
        ext_dir = None
        ext_x = None
        ext_y = None
        ext_origin = None
        ext_dist = None

        for vec in part_vec:
            if vec[0] < 0:
                continue
            cmd_type = int((vec[0] + 0.5).item())
            cmd = ALL_COMMANDS[cmd_type]

            if cmd == 'Line':
                verts.append(vec[1:3])

            elif cmd == 'Arc':
                if len(verts) == 0:
                    raise ValueError("Arc command must be preceded by a Line command.")
                end_point = vec[1:3]
                if not rad:
                    sweep_angle = vec[3] / 180 * torch.pi
                else:
                    sweep_angle = vec[3]
                clock_sign = int(vec[4].item())
                start_point = verts[-1]
                s2e_vec = end_point - start_point
                if s2e_vec.norm() < eps:
                    raise ValueError("Start and end points are the same; cannot define an arc.")
                radius = (s2e_vec.norm() / 2) / (torch.sin(sweep_angle / 2) + eps)
                s2e_mid = (start_point + end_point) / 2
                vertical = torch.stack([s2e_vec[1], -s2e_vec[0]])
                vertical = vertical / (vertical.norm() + eps)
                if clock_sign == 0:
                    vertical = -vertical
                center_point = s2e_mid - vertical * (radius * torch.cos(sweep_angle / 2))
                ref_vec = (end_point - center_point) if clock_sign == 0 else (start_point - center_point)
                ref_vec = ref_vec / (ref_vec.norm() + eps)
                res = 20
                angles = torch.linspace(sweep_angle / res, sweep_angle, res - 1, device=device)
                for angle in angles:
                    rot_cos = torch.cos(angle)
                    rot_sin = torch.sin(angle)
                    rot_matrix = torch.stack([
                        torch.stack([rot_cos, -rot_sin]),
                        torch.stack([rot_sin,  rot_cos])
                    ], dim=0)
                    arc_point = center_point + rot_matrix @ (ref_vec * radius)
                    verts.append(arc_point)

            elif cmd == 'Circle':
                center = vec[1:3]
                radius = vec[5]
                res = 40 + 1
                angles = torch.linspace(0., 2 * torch.pi, res, device=device)[:-1]
                for angle in angles:
                    point = center + torch.stack([torch.cos(angle), torch.sin(angle)]) * radius
                    verts.append(point)

            elif cmd == 'Ext':
                mask = torch.tensor(CMD_ARGS_MASK[cmd_type], dtype=torch.bool, device=device)
                cmd_args = vec[1:][mask]
                # cmd_args = torch.cat([cmd_args[:6], cmd_args[7:8]])
                ext_dir, ext_x = polar_parameterization_inverse_torch(
                    cmd_args[0], cmd_args[1], cmd_args[2]
                )
                ext_y = torch.cross(ext_dir, ext_x, dim=0)
                ext_origin = cmd_args[3:6]
                ext_dist = cmd_args[7]

            else:
                continue

        # triangulate profile
        for i in range(1, len(verts) - 1):
            faces.append(torch.tensor([0, i + 1, i], dtype=torch.int64, device=device))
        faces = torch.stack(faces, dim=0)

        # scale and orient
        verts = torch.stack(verts, dim=0) * size[part_idx, 0]
        verts = (
            verts[:, 0:1] * ext_x.unsqueeze(0)
            + verts[:, 1:2] * ext_y.unsqueeze(0)
            + ext_origin.unsqueeze(0)
            + loc[part_idx].unsqueeze(0)
        )

        # extrude and collect
        offset = ext_dir * ext_dist * size[part_idx, 1]
        verts_ext, faces_ext = extrude_convex_mesh(verts, faces, offset)
        verts_all.append(verts_ext)
        faces_all.append(faces_ext)

    verts_all, faces_all = append_faces(verts_all, faces_all)
    return verts_all, faces_all


def param2sdf(part_names, part_cad_vec, size, loc, query_points, rad=False, verts=None, faces=None):
    if verts is None or faces is None:
        if isinstance(part_cad_vec[0], np.ndarray):
            verts, faces = param2mesh(part_names, part_cad_vec, size, loc)
        else:
            verts, faces = param2mesh_torch(part_names, part_cad_vec, size, loc, rad=rad)

    a = verts[faces[:, 0]]  # N x 3
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    ba = b - a
    cb = c - b
    ac = a - c
    pa = query_points[:, None, :] - a[None, :, :]  # P x N x 3
    pb = query_points[:, None, :] - b[None, :, :]
    pc = query_points[:, None, :] - c[None, :, :]
    nor = torch.cross(ba, ac, dim=-1)  # N x 3
    
    indicator = torch.sign(torch.sum(torch.cross(ba, nor, dim=-1) * pa, dim=-1)) + \
        torch.sign(torch.sum(torch.cross(cb, nor, dim=-1) * pb, dim=-1)) + \
        torch.sign(torch.sum(torch.cross(ac, nor, dim=-1) * pc, dim=-1)) < 1.5  # P x N
    
    sdf = torch.sqrt(torch.where(indicator, torch.minimum(torch.minimum(torch.sum(torch.pow(ba * torch.clamp(torch.sum(ba * pa, dim=-1) / torch.sum(ba * ba, dim=-1), 0, 1)[..., None] - pa, 2), -1),
                                                            torch.sum(torch.pow(cb * torch.clamp(torch.sum(cb * pb, dim=-1) / torch.sum(cb * cb, dim=-1), 0, 1)[..., None] - pb, 2), -1)),
                                               torch.sum(torch.pow(ac * torch.clamp(torch.sum(ac * pc, dim=-1) / torch.sum(ac * ac, dim=-1), 0, 1)[..., None] - pc, 2), -1)),
                      torch.pow(torch.sum(nor * pa, dim=-1), 2) / torch.sum(nor * nor, -1)))  # P x N
    
    return torch.min(sdf, dim=-1)[0]

# %%
import copy
def param2vec(part_names, part_cad_vec, size, loc):
    part_names_res = copy.deepcopy(part_names)
    part_cad_vec_res = copy.deepcopy(part_cad_vec)
    for part_idx, (part_name, part_vec) in enumerate(zip(part_names_res, part_cad_vec_res)):
        for vec in part_vec:
            cmd_type = int(vec[0] + 0.5)
            if ALL_COMMANDS[cmd_type] == 'SOL':
                continue
            elif ALL_COMMANDS[cmd_type] == 'Line' or ALL_COMMANDS[cmd_type] == 'Arc':
                vec[1:3] *= size[part_idx, 0]
            elif ALL_COMMANDS[cmd_type] == 'Circle':
                vec[5] *= size[part_idx, 0]
            elif ALL_COMMANDS[cmd_type] == 'Ext':
                vec[9:12] += loc[part_idx]
                vec[13] *= size[part_idx, 1]
    return part_names_res, part_cad_vec_res
