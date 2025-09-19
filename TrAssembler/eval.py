import torch
import torch.optim as optim
import numpy as np
from cadlib.macro import *
from einops import rearrange
from dataset import collate_fn
from cadlib.visualize import vec2CADsolid
from utils.io import cad_to_obj
from tqdm import tqdm
import clip
from functools import partial
import h5py
import os
from omegaconf import OmegaConf
from model import GMFlowModel
import cv2
from scipy.spatial import KDTree
import trimesh
from pathlib import Path
import json
import sys
from torchvision.transforms import ToTensor, Normalize, Compose
from einops import rearrange
from functools import partial
from dataset import extract_base_name, remove_leading_ending_symbols, replace_digits_with_zero
from glob import glob
import argparse
from utils.misc import render_nvdiffrast_rgb
from utils.diffsdf import param2mesh_torch, param2sdf
import logging
from scipy.spatial.transform import Rotation

torch.set_grad_enabled(False)
# Set the trimesh logger to suppress everything below ERROR
trimesh.util.attach_to_log(level=logging.ERROR)

def center_data_single(pc):
    centroid = np.mean(pc, axis=0)
    pc[:, 0] -= centroid[0]
    pc[:, 1] -= centroid[1]
    pc[:, 2] -= centroid[2]
    return pc


def create_sample(h5, part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model):
    part_cad_vec = []
    for name in part_names:
        part_vec = h5[name]
        while type(part_vec) == h5py._hl.group.Group:
            for k, v in part_vec.items():
                part_vec = v
                break
        part_cad_vec.append(part_vec[:])

    base_part_names = [extract_base_name(remove_leading_ending_symbols(part.replace('#', ' '))) for part in part_names]

    part_name_embeddings = get_clip_features(clip_model, base_part_names, 'cuda').cpu().numpy()  # [n_part, n_dim]

    if do_retrieval:
        # find nearest name in db
        for i in range(len(part_name_embeddings)):
            dist = np.linalg.norm(part_name_embeddings[i] - name_embeddings_db, axis=-1)
            nearest_idx = dist.argmin()
            part_name_embeddings[i] = name_embeddings_db[nearest_idx]
            # part_names[i] = name_db[nearest_idx]
            
    part_names_no_digit = np.array([replace_digits_with_zero(n) for n in part_names])
    part_index = np.zeros(len(part_names))
    for name in np.unique(part_names_no_digit):
        group_idx = np.where(part_names_no_digit == name)[0]
        part_index[group_idx] = np.arange(len(group_idx))


    sample = {
        "part_cad": part_cad_vec,
        "part_name_emb": part_name_embeddings,
        "part_index": part_index,
        "img_tex": image,
        "img_non_tex": image,
        "data_id": data_id,
    }
    
    return sample


def normalize_data_single(pc):
    # get furthest point distance then normalize
    d = max(np.sum(np.abs(pc)**2, axis=-1)**(1./2))
    pc /= d

    return pc


def chamfer_distance_kdtree(pc1, pc2):
    """
    Compute the Chamfer Distance between two point clouds using KD-trees.

    Parameters:
    - pc1: Numpy array of shape (N, D) representing the first point cloud.
    - pc2: Numpy array of shape (M, D) representing the second point cloud.

    Returns:
    - distance: The Chamfer Distance between the two point clouds.
    """
    # Create KD-trees for each point cloud
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)

    # For each point in pc1, find its nearest neighbor in pc2
    distances1, _ = tree1.query(pc2)
    # For each point in pc2, find its nearest neighbor in pc1
    distances2, _ = tree2.query(pc1)

    # The Chamfer Distance is the sum of both components
    return np.mean(distances1) + np.mean(distances2)


def compute_cd(mesh1, mesh2, normalize=True):
    n_points = 2048
    # Suppress trimesh sampling messages
    pts1 = trimesh.sample.sample_surface_even(mesh1, n_points)[0]
    pts2 = trimesh.sample.sample_surface_even(mesh2, n_points)[0]

    # if number of points is less than 2048, randomly sample points for remaining points
    if len(pts1) < n_points:
        pts1 = np.concatenate([pts1, mesh1.sample(n_points - len(pts1))])
    
    if len(pts2) < n_points:
        pts2 = np.concatenate([pts2, mesh2.sample(n_points - len(pts2))])
    
    if normalize:
        pts1 = center_data_single(pts1)
        pts1 = normalize_data_single(pts1)
        pts2 = center_data_single(pts2)
        pts2 = normalize_data_single(pts2)
    cd = chamfer_distance_kdtree(pts1, pts2)
    return cd


def symmetry_opt(gmflow, x_t, t, cmd, cmd_mask, args_mask, gt_arc_arg, model: GMFlowModel, sym_type: str):
    # print(t)
    
    if sym_type == 'no_symmetry':
        return x_t
    
    num_iter = int(np.sqrt(1000 - t.item())) - 20
    if num_iter <= 0:
        return x_t
    has_arc = False
    x_t = rearrange(x_t.clone(), 'B N P L -> B P L N')
    model.denormalize(x_t, args_mask)
    x_t = x_t[0].clone()
    cmd = cmd[0].clone()
    cmd_mask = cmd_mask[0].clone()
    # print(x_t.shape, cmd.shape, cmd_mask.shape, t)
    p_indices, l_indices = torch.nonzero(cmd == 5, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 0:5] = -1
        x_t[p_indices, l_indices, -3:] = 0
        x_t[p_indices, l_indices, -5] = 1.
    p_indices, l_indices = torch.nonzero(cmd == 4, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices] = -1
    p_indices, l_indices = torch.nonzero(cmd == 0, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:] = -1
    p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
    if p_indices.shape[0] > 0:
        has_arc = True
        x_t[p_indices, l_indices, 4:] = -1
        if gt_arc_arg is not None:
            x_t[p_indices, l_indices, 3] = gt_arc_arg[3]
        else:
            x_t[p_indices, l_indices, 3] = 1.
        x_t[p_indices, l_indices, 2] *= 180 / np.pi
    p_indices, l_indices = torch.nonzero(cmd == 2, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:4] = -1
        x_t[p_indices, l_indices, 5:] = -1
    cmd[cmd_mask == 0] = -1

    x_t = x_t.detach().requires_grad_(True)
    P, L, N = x_t.shape
    optimizer = optim.Adam([x_t], lr=1e-4)
    # bbox_min, bbox_max = mesh.bounds
    # bbox_min = torch.from_numpy(bbox_min).cuda()
    # bbox_max = torch.from_numpy(bbox_max).cuda()
    # part_vecs_torch = part_vecs_torch.clone().detach().requires_grad_(True)
    size = torch.ones((P, 2), dtype=torch.float32, device='cuda', requires_grad=False)
    loc = torch.zeros((P, 3), dtype=torch.float32, device='cuda', requires_grad=False)
    
    # bbox_min = torch.full((3,), -0.5, dtype=torch.float32, device='cuda', requires_grad=False)
    # bbox_max = torch.full((3,), 0.5, dtype=torch.float32, device='cuda', requires_grad=False)

    with torch.enable_grad():
        # Run optimization for 100 iterations
        for iter in range(num_iter):
            part_vecs = torch.cat([cmd[..., None], x_t], -1)

            # Compute SDF for original and reflected points
            verts, faces = param2mesh_torch([None] * P, part_vecs, size, loc, rad=False)
            if iter == 0:
                bbox_min = torch.min(verts, dim=0)[0].detach()
                bbox_max = torch.max(verts, dim=0)[0].detach()
            
            # Sample query points with x < 0 within the bounding box
            query_points = torch.rand((1024, 3), device='cuda') * (bbox_max - bbox_min) + bbox_min
            query_points = query_points[query_points[:, 0] < 0]  # Keep only points with x < 0
            
            sdf_original = param2sdf([None] * P, part_vecs, size, loc, query_points, rad=False, verts=verts, faces=faces)
            if sym_type == 'reflection':
                            # Reflect the query points along the x=0 plane
                reflected_points = query_points.clone()
                reflected_points = reflected_points * torch.tensor([-1.,1,1], device=reflected_points.device)
                sdf_reflected = param2sdf([None] * P, part_vecs, size, loc, reflected_points, rad=False, verts=verts, faces=faces)
                symmetry_loss = torch.mean(torch.abs(sdf_original - sdf_reflected))
            elif sym_type == 'rotational':
                bins = 10
                symmetry_loss = 0.
                for i in range(bins):
                    rot_around_y = Rotation.from_euler('y', 360 * i / bins, degrees=True).as_matrix()
                    rotated_points = query_points.clone()
                    rotated_points = (rotated_points @ rot_around_y.T).T
                    sdf_reflected = param2sdf([None] * P, part_vecs, size, loc, rotated_points, rad=False, verts=verts, faces=faces)
                    symmetry_loss += torch.mean(torch.abs(sdf_original - sdf_reflected))
                symmetry_loss /= bins
            else:
                raise ValueError(f"Invalid symmetry type: {sym_type}")
            # Compute symmetry loss as the mean absolute difference between the two SDFs
            # print(f"Iteration {iter}, Symmetry Loss: {symmetry_loss.item()}")

            # Backpropagation and optimization step
            optimizer.zero_grad()
            symmetry_loss.backward()
            if has_arc:  # no optimization for arc
                p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
                x_t.grad[p_indices, l_indices, 2] = 0.
            optimizer.step()
            
    x_t = x_t.detach().clone()
    p_indices, l_indices = torch.nonzero(cmd == 5, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 0:5] = 0.
        x_t[p_indices, l_indices, -3:] = 0.
        x_t[p_indices, l_indices, -5] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 4, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 0, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 2, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 2:4] = 0.
        x_t[p_indices, l_indices, 5:] = 0.
    p_indices, l_indices = torch.nonzero(cmd == 1, as_tuple=True)
    if p_indices.shape[0] > 0:
        x_t[p_indices, l_indices, 4:] = 0.
        x_t[p_indices, l_indices, 3] = 0.
        x_t[p_indices, l_indices, 2] /= 180 / np.pi
    x_t = x_t[None]
    # print(x_t[0, p_indices, l_indices, 2])
    model.normalize(x_t, args_mask)
    # print(x_t[0, p_indices, l_indices, 2])
    return rearrange(x_t, 'B P L N -> B N P L')


def step_callback(
        gmflow,
        x_t,
        t,
        cmd,
        cmd_mask,
        args_mask,
        gt_arc_arg,
        model: GMFlowModel,
        save_intermediate=False,
        output_subdir: Path = None,
        part_names=None,
        base_args: torch.Tensor = None,
        rgb_gt: np.ndarray = None,
        sym_type: str = None,
    ):
    """Wrapper around symmetry_opt that optionally saves intermediate results per timestep.

    Saves to output_subdir/intermediate/step_XXXX/{intermediate.h5, intermediate.obj, intermediate.png}.
    """
    # Run the optimization step first
    x_t_new = symmetry_opt(gmflow, x_t, t, cmd, cmd_mask, args_mask, gt_arc_arg, model, sym_type)

    if not save_intermediate or output_subdir is None:
        return x_t_new

    try:
        step_id = int(t.item()) if torch.is_tensor(t) else int(t)
    except Exception:
        step_id = 0

    try:
        inter_root = output_subdir / 'intermediate'
        step_dir = inter_root / f'step_{step_id:04d}'
        step_dir.mkdir(parents=True, exist_ok=True)

        # Prepare current denormalized args for conversion to CAD vectors
        x_denorm = rearrange(x_t_new.clone(), 'B N P L -> B P L N')
        model.denormalize(x_denorm, args_mask)

        # Merge with base args (keep non-predicted values from base)
        if base_args is not None:
            args_current = base_args.clone()
            mask = args_mask[0] > 0
            args_current[mask] = x_denorm[0][mask]
        else:
            args_current = x_denorm[0]

        # Build CAD vectors
        cad_vec = []
        part_cad_vec_out = {}
        # import pdb; pdb.set_trace()
        _, P, L = cmd.shape
        for p in range(P):
            if cmd_mask[0][p][0] > 0:
                part_vec = []
                for l in range(L):
                    if cmd_mask[0][p][l] > 0:
                        arg = args_current[p][l].detach().cpu().numpy()
                        cmd_type = int(cmd[0][p][l].item())
                        if cmd_type == 1:
                            arg[2] /= np.pi / 180.
                            arg[3] = 1.0
                        cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
                        part_vec.append(cad_vec[-1])
                if part_names is not None and p < len(part_names):
                    part_cad_vec_out[part_names[p]] = np.array(part_vec)

        cad_vec_np = np.array(cad_vec)

        # Save h5
        with h5py.File(str(step_dir / 'intermediate.h5'), 'w') as h5_out:
            h5_out.create_dataset('vec', data=cad_vec_np)
            for k in part_cad_vec_out.keys():
                h5_out.create_dataset(k, data=part_cad_vec_out[k], dtype=float)

        # Save obj and png
        full_shape = vec2CADsolid(cad_vec_np, is_numerical=False)
        obj_path = step_dir / 'intermediate.obj'
        cad_to_obj(full_shape, str(obj_path))
        if rgb_gt is not None:
            rgb_pred = render_nvdiffrast_rgb(str(obj_path))
            rgb = np.concatenate([rgb_gt, rgb_pred], axis=1)
            cv2.imwrite(str(step_dir / 'intermediate.png'), rgb[..., ::-1])

    except Exception as e:
        # Never break sampling due to intermediate saving errors
        print(f"Error saving intermediate: {e}")
        import traceback
        print(traceback.format_exc())

    return x_t_new

def get_clip_features(clip_model, texts, device):
    """Get CLIP text features for given texts."""
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate TrAssembler model.")
    parser.add_argument("--model_dir", type=str, default='data/ckpts/trassembler/chair', help="Model directory.")
    parser.add_argument("--output_dir", type=str, default='data/output/trassembler_h5', help="Output directory.")
    parser.add_argument("--text_emb_retrieval", action='store_true', help="Use text embedding retrieval.")
    parser.add_argument("--data_root", type=str, default='data', help="Data root directory.")
    parser.add_argument("--skip_processed", action='store_true', help="Skip processed objects.")
    parser.add_argument("--sym_labels_dir", type=str, default='data/sym_labels', help="Symmetry label directory.")
    parser.add_argument("--save_intermediate", action='store_true', help="Save intermediate step outputs (h5, obj, png) per timestep under out_dir/data_id/intermediate.")
    
    args = parser.parse_args()
    save_intermediate = args.save_intermediate
    skip_processed = args.skip_processed
    sym_labels_dir = args.sym_labels_dir
    ckpt_path = Path(f'{args.model_dir}/checkpoints/last.ckpt')
    config = OmegaConf.load(os.path.join(args.model_dir, '.hydra/config.yaml'))
    model = GMFlowModel.load_from_checkpoint(ckpt_path,
                                                   args=config, embed_dim=config.network.embed_dim, num_heads=config.network.num_heads, dropout=config.network.dropout,
                                  bias=True, scaling_factor=1., args_range=np.array([-1., 1.])).cuda().eval()

    image_transform = Compose(
        [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    
    args.output_dir = os.path.join(args.output_dir, config.category)


    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    name_db = list(set(json.load(open(f'data/partnet2common_{config.category}.json')).values()))
    name_db = [name[:name.find('/')] if '/' in name else name for name in name_db]
    name_embeddings_db = clip_model.encode_text(clip.tokenize(name_db).to('cuda')).cpu().numpy()
    name_embeddings_db /= np.linalg.norm(name_embeddings_db, axis=1, keepdims=True)
    print(f"Name embeddings shape: {name_embeddings_db.shape}")
    print(f"Name database: {name_db}")

    np.random.seed(0)
    do_retrieval = args.text_emb_retrieval
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    test_ids = open(f'data/splits/{config.category}_test_ids.txt').read().splitlines()
    test_ids = [int(i) for i in test_ids]

    h5_root = Path(f'data/output/llamaft_h5/{config.category}')
    gt_data_root = Path(f'data/raw_annotated/{config.category}')

    occs = []
    cds = []
    cnt = 0
    
    sym_label2type = {1: 'rotational', 2: 'reflection', 3: 'no_symmetry'}
    for data_id in tqdm(test_ids):
        sym_label_path = f'{sym_labels_dir}/{data_id}.txt'
        if not os.path.exists(sym_label_path):
            print(f'{sym_label_path} not exists')
            occs.append(0.)
            continue
        sym_label = open(sym_label_path).read().splitlines()[0]
        sym_type = sym_label2type[int(sym_label)]
        # Prepare per-object output directory
        obj_out_dir = out_dir / f'{data_id}'
        obj_out_dir.mkdir(exist_ok=True, parents=True)

        # if already processed, skip
        final_obj_path = obj_out_dir / 'final.obj'
        if os.path.exists(str(final_obj_path)) and skip_processed:
            obj_path = str(final_obj_path)
            occs.append(1.)

            gt_mesh_path = f'{gt_data_root}/{data_id}/raw.obj'
            if os.path.exists(gt_mesh_path):
                gt_mesh = trimesh.load_mesh(gt_mesh_path)
                pred_mesh = trimesh.load_mesh(obj_path)
                cd = compute_cd(gt_mesh, pred_mesh, normalize=True)
                cds.append(cd)
            continue
        h5_path = h5_root / f'{data_id}.h5'
        if not h5_path.exists():
            print(f'{h5_path} not exists')
            occs.append(0.)
            continue
        h5 = h5py.File(h5_path, 'r')
        part_names = list(sorted([n for n in h5 if n != 'vec' and '_bbox' not in n]))  # this has no effect, h5 will treat / as hierarchy

        gt_h5_path = f'data/raw_annotated/{config.category}/{data_id}/cad.h5'
        gt_h5 = h5py.File(gt_h5_path, 'r')
        gt_part_names = list(sorted([n for n in gt_h5 if n != 'vec' and '_bbox' not in n]))  # this has no effect, h5 will treat / as hierarchy
        cnt += 1


        image_path = f'data/blender_renderings/{data_id}.png'
        image = cv2.imread(image_path)[..., ::-1].copy()
        image = image_transform(image)
        rgb_gt = cv2.imread(image_path)[..., ::-1].copy()

        sample = create_sample(h5, part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model)
        gt_sample = create_sample(gt_h5, gt_part_names, data_id, image, do_retrieval, name_embeddings_db, clip_model)

        batch = collate_fn([sample])
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
                
        gt_batch = collate_fn([gt_sample])
        for k in gt_batch:
            if isinstance(gt_batch[k], torch.Tensor):
                gt_batch[k] = gt_batch[k].cuda()
        gt_args = gt_batch['args'][0].clone().detach()
        gt_cmd = gt_batch['command'][0].clone().detach()
        arc_cmd = gt_cmd == 1
        if arc_cmd.any():
            gt_arc_arg = gt_args[arc_cmd][0]
        else:
            gt_arc_arg = None

        try:
            step_cb = partial(
                step_callback,
                cmd=batch['command'],
                cmd_mask=batch['command_mask'],
                args_mask=batch['args_mask'],
                gt_arc_arg=gt_arc_arg,
                model=model,
                save_intermediate=save_intermediate,
                output_subdir=obj_out_dir,
                part_names=part_names,
                base_args=batch['args'][0].clone().detach(),
                rgb_gt=rgb_gt,
                sym_type=sym_type,
            )
            pred_args = model.sample(batch, step_callback=step_cb)
        except Exception as e:
            print(f"Error during sampling: {e}")
            occs.append(0.)
            continue
        # Denormalize the predicted args
        model.denormalize(pred_args, batch['args_mask'])

        # Assuming batch size is always 1 for evaluation
        i = 0 # Since batch size is 1
        data_id = batch['data_id'][i] # Use the actual data_id from the batch
        # rgb_gt already loaded above

        cad_vec = []
        cmd = batch['command'][i]  # [P, L]
        cmd_mask = batch['command_mask'][i]  # [P, L]
        arg_mask = batch['args_mask'][i]  # [P, L, N]
        args = batch['args'][i]  # [P, L, N]
        args[arg_mask > 0] = pred_args[i][arg_mask > 0]  # overwrite the args with predicted args

        part_cad_vec_out = {} # Renamed to avoid conflict
        for p in range(cmd_mask.shape[0]):
            if cmd_mask[p][0] > 0:  # this part is valid
                part_vec = []
                for l in range(cmd_mask.shape[1]):
                    if cmd_mask[p][l] > 0:
                        arg = args[p][l].cpu().numpy()
                        cmd_type = cmd[p][l].item()
                        if cmd_type == 1: # ARC
                            # Ensure angle is within valid range if needed, though denormalization should handle it
                            arg[2] /= np.pi / 180. # Convert back to degrees if needed by vec2CADsolid
                            arg[3] = 1.0
                        cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
                        part_vec.append(cad_vec[-1])

                # Use original part names from the loaded h5 file
                part_cad_vec_out[part_names[p]] = np.array(part_vec)
        cad_vec = np.array(cad_vec)
        try:
            full_shape = vec2CADsolid(cad_vec, is_numerical=False)
            # save obj
            obj_path = str(final_obj_path)
            cad_to_obj(full_shape, obj_path)
            rgb_pred = render_nvdiffrast_rgb(obj_path)
            # vis.image(np.moveaxis(rgb, -1, 0), win='val', opts={'title': 'val'})
            occs.append(1.)

            gt_mesh_path = f'{gt_data_root}/{data_id}/raw.obj'
            if os.path.exists(gt_mesh_path):
                gt_mesh = trimesh.load_mesh(gt_mesh_path)
                pred_mesh = trimesh.load_mesh(obj_path)
                cd = compute_cd(gt_mesh, pred_mesh, normalize=True)
                cds.append(cd)
            
        except Exception as e:
            print(f"Error processing {data_id}: {e}")
            rgb_pred = np.zeros((400, 400, 3), dtype=np.uint8)
            occs.append(0.)
            continue
        rgb = np.concatenate([rgb_gt, rgb_pred], axis=1)

        cv2.imwrite(str(obj_out_dir / 'final.png'), rgb[..., ::-1])

        # save h5 and the obj
        with h5py.File(str(obj_out_dir / 'final.h5'), 'w') as h5_out: # Renamed handle
            h5_out.create_dataset('vec', data=cad_vec)
            for part_name in part_cad_vec_out.keys():
                h5_out.create_dataset(part_name, data=part_cad_vec_out[part_name], dtype=float)

    # Calculate metrics ignoring NaNs
    valid_cds = [c for c in cds if not np.isnan(c)]
    mean_occ = np.mean(occs) if occs else 0.0
    mean_cd = np.mean(valid_cds) if valid_cds else float('nan')

    print(f'Total processed: {len(test_ids)}')
    print(f'Successful generations (OCC > 0): {sum(occs)}')
    print(f'Valid CD computations: {len(valid_cds)}')
    print(f'Mean OCC: {mean_occ:.4f}')
    print(f'Mean CD (valid only): {mean_cd:.4f}')

if __name__ == "__main__":
    main()