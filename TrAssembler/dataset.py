from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import cv2
from cadlib.macro import *
from cadlib.extrude import CADSequence
from copy import deepcopy
import pickle
from cadlib.math_utils import polar_parameterization_inverse, polar_parameterization
from torchvision.transforms import ToTensor, Normalize, Compose
from tqdm import tqdm
import clip
import re
import numpy as np
from pathlib import Path


def remove_leading_ending_symbols(text):
    text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', text)
    return text.strip()


def extract_base_name(text):
    # Find the first occurrence of a number, considering any non-digit characters (excluding spaces) before it
    match = re.search(r'(.*)(\s+)\D*(\d+)', text)
    if match:
        # Extract the text before the last consecutive spaces before the number
        name_part = match.group(1).strip()
    else:
        # If no number is found, use the whole text
        name_part = text.strip()
    # to lowercase
    name_part = name_part.lower()
    return name_part

def stabilize(v, eps):
    n, dim = v.shape
    for d in range(dim):
        for ii in range(n):
            for jj in range(n):
                if abs(v[ii][d] - v[jj][d]) < eps:
                    v[jj][d] = v[ii][d]
    return v


def rotate_vector(v, axis, angle):
    """
    Rotate a vector v around an axis by a given angle in degrees.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)
    
    # Normalize the axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    
    # Rodrigues' rotation formula
    axis_cross_v = np.cross(axis, v)
    axis_dot_v = np.dot(axis, v)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Compute rotated vector
    rotated_v = cos_angle * v + sin_angle * axis_cross_v + (1 - cos_angle) * axis_dot_v * axis
    return rotated_v


def get_mid_point(start_point, end_point, sweep_angle, clock_sign):
    s2e_vec = end_point - start_point
    if np.linalg.norm(s2e_vec) < 1e-6:
        return start_point

    radius = (np.linalg.norm(s2e_vec) / 2) / np.sin(sweep_angle / 2)            

    s2e_mid = (start_point + end_point) / 2
    vertical = np.cross(s2e_vec, [0, 0, 1])[:2]
    vertical = vertical / (np.linalg.norm(vertical) + 1e-8)
    if clock_sign == 0:
        vertical = -vertical
    center = s2e_mid - vertical * (radius * np.cos(sweep_angle / 2))

    start_angle = 0
    end_angle = sweep_angle
    if clock_sign == 0:
        ref_vec = end_point - center
    else:
        ref_vec = start_point - center
    ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)

    #######
    mid_angle = (start_angle + end_angle) / 2
    rot_mat = np.array([[np.cos(mid_angle), -np.sin(mid_angle)],
                        [np.sin(mid_angle), np.cos(mid_angle)]])
    mid_vec = rot_mat @ ref_vec
    mid_point = center + mid_vec * radius
    #######
    return mid_point


def compute_aabbox_2d(vertices):
    min_x = np.min(vertices[:,0])
    max_x = np.max(vertices[:,0])
    min_y = np.min(vertices[:,1])
    max_y = np.max(vertices[:,1])

    double_s = np.array([max_x-min_x, max_y-min_y])
    c = np.array([min_x+double_s[0]/2.0, min_y+double_s[1]/2.0])
    ax = np.array([1.,0.], dtype=np.float32)
    ay = np.array([0.,1.], dtype=np.float32) 

    aabbox = np.zeros(12, dtype=np.float32)
    aabbox[0:2] = c
    aabbox[2:4] = double_s
    aabbox[4:6] = ax
    aabbox[6:8] = ay

    return aabbox


def compute_sketch_center(cmds):
    all_points = []
    for cmd in cmds:
        if int(cmd[0] + .5) in [0, 1]:
            all_points.append(cmd[1:3])
        if int(cmd[0] + 0.5) == 2:
            all_points.append(cmd[1:3])

    ####### Handle arcs --> add the center point of the arc
    all_curve_idxs = [i for i, cmd in enumerate(cmds) if int(cmd[0] + .5) == 1]

    for idx in all_curve_idxs:
        cmd = cmds[idx]
        #### Get the start point from the previous command if line or arc
        if idx != 0 and int(cmds[idx-1][0] + .5) in [0, 1]:
            start_point = np.array(cmds[idx-1][1:3])
        ### else it is the origin
        else:
            start_point = np.array([0., 0.])

        end_point = np.array(cmd[1:3])
        sweep_angle = cmd[3] / 180 * np.pi
        # cmds[idx][4] = 0
        clock_sign = int(cmd[4] + .5)
        # clock_sign = 0

        mid_point = get_mid_point(start_point, end_point, sweep_angle, clock_sign)

        all_points.append(mid_point)
    #####################################
    if len(all_points) == 0:
        return None
    # cmd_center = np.mean(all_points, axis=0)
    cmd_center = compute_aabbox_2d(np.unique(np.array(all_points).round(decimals=2), axis=0))[0:2]
    return cmd_center


def collate_fn(data):
    """
    Prepare batch data for processing by padding variable-length CAD data.
    
    This function handles the complex task of batching CAD data with variable numbers
    of parts and lines per part. It pads sequences to create uniform tensor shapes
    and creates appropriate masks for training.
    
    Args:
        data (list): List of dictionaries with keys:
            - 'part_cad': List of CAD command vectors per part
            - 'part_name_emb': CLIP embeddings of part names  
            - 'img_tex': Textured image tensor
            - 'part_index': Part ordering indices
            - 'data_id': Unique identifier for the sample
    
    Returns:
        dict: Batched tensors with keys:
            - 'command': Command type indices [B, P, L]
            - 'command_mask': Valid command mask [B, P, L] 
            - 'args': Continuous parameters [B, P, L, N]
            - 'args_mask': Valid argument mask [B, P, L, N]
            - 'part_name_embedding': Part semantic embeddings [B, P, D]
            - 'part_index': Part ordering [B, P]
            - 'img_tex': Input images [B, C, H, W]
            - 'data_id': Sample identifiers [B]
    """
    # import pdb; pdb.set_trace()
    max_N_part = max(len(d['part_cad']) for d in data)
    max_N_lines = max(max(len(part) for part in d['part_cad']) for d in data)

    def pad_sequences(sequences, max_len, pad_value=0, dtype=np.float32):
        """ Pad a list of sequences to the same length with pad_value. """
        padded = np.full((len(sequences), max_len, *sequences[0].shape[1:]), pad_value, dtype=dtype)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded

    batch_data = {
        'padded_cad': [], 'command_mask': [], 'part_name_emb': [],
        'part_index': [], 'img_tex': [], 'img_non_tex': [], 'data_id': [], 'args_mask': []
    }
    
    for item in data:
        part_cads = item['part_cad']
        part_name_embs = item['part_name_emb']
        
        padded_cads = [np.concatenate([cad, EOS_VEC[np.newaxis].repeat(max_N_lines - len(cad), axis=0)]) for cad in part_cads]
        masks = [np.concatenate([np.ones(len(cad), dtype=int), np.zeros(max_N_lines - len(cad), dtype=int)]) for cad in part_cads]

        if len(padded_cads) < max_N_part:
            extra_padding = max_N_part - len(padded_cads)
            padded_cads += [EOS_VEC[np.newaxis].repeat(max_N_lines, axis=0)] * extra_padding
            masks += [np.zeros(max_N_lines, dtype=int)] * extra_padding
            part_name_embs = np.concatenate([part_name_embs, np.zeros((extra_padding, part_name_embs.shape[-1]))])
            item['part_index'] = np.concatenate([item['part_index'], np.zeros(extra_padding)])

        padded_cads = np.stack(padded_cads)
        args_mask = (np.abs(padded_cads[..., 1:] + 1) > 1e-4).astype(int)
        ext_mask = (padded_cads[..., 0] + 0.5).astype(int) == 5
        args_mask[ext_mask, -3:] = 0  # mask extrusion boolean type
        args_mask[ext_mask, -5] = 0 # size is always 1
        padded_cads[ext_mask, -5] = 1.0  # size is always 1
        
        # convert from angle to rad
        padded_cads[(padded_cads[..., 0] + 0.5).astype(int) == 1, 3] *= np.pi / 180.
        
        # mask the counter clockwise flag
        args_mask[(padded_cads[..., 0] + 0.5).astype(int) == 1, 3] = 0
        
        batch_data['args_mask'].append(args_mask)
        batch_data['padded_cad'].append(padded_cads)  # [N_part, max_N_lines, part_vec_dim]
        batch_data['command_mask'].append(np.stack(masks))  # [N_part, max_N_lines]
        batch_data['part_name_emb'].append(part_name_embs)  # [N_part, emb_dim]
        batch_data['part_index'].append(item['part_index'])  # [N_part]
        batch_data['img_tex'].append(item['img_tex'])
        batch_data['img_non_tex'].append(item['img_non_tex'])
        batch_data['data_id'].append(item['data_id'])

    # Convert lists to tensors
    for key in ['padded_cad', 'command_mask', 'part_name_emb', 'part_index', 'args_mask']:
        batch_data[key] = torch.tensor(np.stack(batch_data[key]), dtype=torch.float32)

    batch_data['img_tex'] = torch.tensor(np.stack(batch_data['img_tex']), dtype=torch.float32)
    batch_data['img_non_tex'] = torch.tensor(np.stack(batch_data['img_non_tex']), dtype=torch.float32)
    batch_data['data_id'] = np.stack(batch_data['data_id'])
    
    args = batch_data['padded_cad'][..., 1:]
    args_mask = batch_data['args_mask']

    return {
        'command': batch_data['padded_cad'][..., 0].round().long(),
        'command_mask': batch_data['command_mask'],
        'args': args,
        'args_mask': args_mask,
        'part_name_embedding': batch_data['part_name_emb'],
        'part_index': batch_data['part_index'],
        'img_tex': batch_data['img_tex'],
        'img_non_tex': batch_data['img_non_tex'],
        'data_id': batch_data['data_id']
    }


def compute_bounding_box(obj_file):
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')

    with open(obj_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:])
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_z = min(min_z, z)
                max_z = max(max_z, z)

    return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                            num_workers=1,
                            collate_fn=collate_fn,
                            # num_workers=config.num_workers,
                            # worker_init_fn=np.random.seed()
                            )
    return dataloader


def replace_digits_with_zero(input_string):
    result = ''.join('0' if char.isdigit() else char for char in input_string)
    return result


class DiskCADDataset(Dataset):
    def __init__(self, directory='dataset', white_list=None):
        super().__init__()
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
        self.data_ids = [int(f.split('/')[-1].split('.')[0]) for f in self.files]
        self.data = {}
        # import pdb; pdb.set_trace()
        if white_list is not None:
            self.data_ids = [data_id for data_id in self.data_ids if data_id in white_list]
            self.files = [f for f in self.files if int(f.split('/')[-1].split('.')[0]) in white_list]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        
        if not os.path.exists(self.files[index]):
            return self.__getitem__((index + 1) % len(self))
        
        with open(self.files[index], 'rb') as f:
            item = pickle.load(f)
        return item



class CADDataset(Dataset):
    """Dataset class for loading and processing CAD data.
    
    This dataset handles loading raw annotated CAD data, processing part semantics,
    normalizing coordinate systems, and preparing data for training.
    
    Args:
        cat: Category name ('chair', 'table', 'storagefurniture')
        data_ids: List of data IDs to include (None for all)
        n_bins: Number of quantization bins (for legacy support)
    """
    def __init__(self, cat='chair', data_ids=None, n_bins=256):
        super().__init__()
        self.image_transform = Compose(
            [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        self.raw_data = f'data/raw_annotated/{cat}'
        self.cat = cat
        if data_ids is None:
            self.all_data = Path(self.raw_data).glob('*/cad.h5')
            self.all_data = [int(p.parent.name) for p in self.all_data]
        
        self.all_data = data_ids
        self.n_bins = n_bins
        self.clip_model, _ = clip.load("ViT-B/32", device='cuda')


    def dump_data_to_disk(self, folder):
        for index in tqdm(list(range(len(self)))[:]):
            data = self.__getitem__(index)
            if data is not None:
                with open(folder / f'{data["data_id"]}.pkl', 'wb') as f:
                    pickle.dump(data, f)
            else:
                if (folder / f'{self.all_data[index]}.pkl').exists():
                    os.remove(folder / f'{data["data_id"]}.pkl')
                
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        data_id = self.all_data[index]
        # data_id = '47917'
        h5 = h5py.File(os.path.join(self.raw_data, str(data_id), "cad.h5"), 'r')
        obj_path = os.path.join(self.raw_data, str(data_id), "cad.obj")
        part_names = list(sorted([n for n in h5 if n != 'vec' and '_bbox' not in n]))[:]

        part_cad_vec = [h5[n][:] for n in part_names]
        
        # use opengl ground-truth rendering
        image_path_textured = os.path.join(f'data/blender_renderings', f'{data_id}.png')
        assert os.path.exists(image_path_textured)
        
        image_textured = cv2.imread(image_path_textured)[..., ::-1].copy()
        image_textured = self.image_transform(image_textured)
        
        # fix axis ambiguity
        for p in part_cad_vec:
            ext_mask = np.where(np.isclose(p[:, 0], 5))[0]
            for i in ext_mask:
                ext_axis, axis_x = polar_parameterization_inverse(*p[i, 6:9])
                ext_loc = p[i, 9:12].copy()
                best_axis = np.abs(ext_axis).argmax()
                if ext_axis[best_axis] < 0:
                    ext_loc += ext_axis * p[i, -4]
                    ext_axis = -ext_axis
                    axis_x = -axis_x
                p[i, 6:9] = polar_parameterization(ext_axis, axis_x)
                p[i, 9:12] = ext_loc
            
        for idx, p in enumerate(part_cad_vec):
            loop_start_index = np.where(np.isclose(p[:, 0], 4))[0][1:]
            loop_cad = np.split(p, loop_start_index)
            loop_cad_cuts = []
            anchors = []
            outer_loop = None
            
            loop_to_delete = []
            for loop_id, loop in enumerate(loop_cad):
                # get the lines
                lines = loop[1:-1]  # the first is SOL, the last is EXT
                ext_cmd = loop[-1]
                ext_axis, axis_x = polar_parameterization_inverse(*ext_cmd[6:9])
                xy = lines[:, 1:3]
                
                # do in plane rotation and find the best one
                cand_axis_xs = np.array([rotate_vector(axis_x, ext_axis, angle) for angle in range(0, 360, 90)])
                
                # remove ambiguity
                best_idx = cand_axis_xs[:, 0].argmax()
                
                # incase the sketch plane is perpendicular to x axis
                if np.abs(cand_axis_xs[best_idx, 0]) < 0.03:
                    best_idx = cand_axis_xs[:, 1].argmax()
                axis_x = cand_axis_xs[best_idx]
                loop[-1, 6:9] = polar_parameterization(ext_axis, axis_x)
                
                sketch_center = compute_sketch_center(lines)
                if sketch_center is None:
                    # this loop is invalid, we should delete it
                    loop_to_delete.append(np.split(np.arange(len(p)), loop_start_index)[loop_id])
                    continue
                
                # rotate the sketch to the best orientation
                rot_mat = np.array([[np.cos(np.radians(best_idx * 90)), -np.sin(np.radians(best_idx * 90))],
                                    [np.sin(np.radians(best_idx * 90)), np.cos(np.radians(best_idx * 90))]])
                loop[1:-1, 1:3] = (loop[1:-1, 1:3] - sketch_center) @ rot_mat.T + np.dot(rot_mat, sketch_center)
                
                xy_sorted = deepcopy(xy)
                xy_stable = deepcopy(xy)
                xy_stable = stabilize(xy_stable, eps=0.03)
                sorted_ids = sorted(list(range(len(xy_stable))),
                                    key=lambda x: (xy_stable[x][0], xy_stable[x][1]))
                cycle = np.array([(sorted_ids[0] + i) % len(xy) for i in range(len(xy))])
                
                xy_sorted = xy_sorted[cycle]

                
                curves_sorted = deepcopy(loop[1:-1, :5])
                curves_sorted = curves_sorted[cycle]
                loop[1:-1, :5] = curves_sorted
                
                if loop[-1][-2] == 0:  # the outer loop
                    assert outer_loop is None
                    outer_loop = loop
                else:
                    loop_cad_cuts.append(loop)
                    anchors.append(xy_sorted[0])
            # then sort the loops based on the anchor point
            if len(anchors) > 0:
                anchors = np.array(anchors)
                anchors = stabilize(anchors, eps=0.03)
                sorted_ids = sorted(list(range(len(anchors))),
                                    key=lambda x: (anchors[x][0], anchors[x][1]))
            else:
                sorted_ids = []
            sorted_loop_cad_cuts = []
            for ori_id, sorted_id in enumerate(sorted_ids):
                sorted_loop_cad_cuts.append(loop_cad_cuts[sorted_id])
            sorted_loop_cad_cuts = [outer_loop] + sorted_loop_cad_cuts
            sorted_loop_cad_cuts = np.concatenate(sorted_loop_cad_cuts, axis=0)
            part_cad_vec[idx] = sorted_loop_cad_cuts
            
            if len(loop_to_delete) > 0:
                for j in np.concatenate(loop_to_delete):
                    np.delete(part_cad_vec[idx], j, axis=0)
                    
            assert sorted_loop_cad_cuts.shape == part_cad_vec[idx].shape

        # sort within each group
        part_index = np.zeros(len(part_names))
        part_cad_vec_sorted = deepcopy(part_cad_vec)
        # find the part with the same name and sort
        # part_names_no_digit = np.array([replace_digits_with_zero(n) for n in part_names])
        base_part_names = [extract_base_name(remove_leading_ending_symbols(part.replace('#', ' '))) for part in part_names]
        
        text = clip.tokenize(base_part_names).to('cuda')
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        part_name_embeddings = text_features.cpu().numpy()  # [n_part, n_dim]
        for name in np.unique(base_part_names):
            group_idx = np.where(np.array(base_part_names) == name)[0]
            # part_names_group = part_names_no_digit[group_idx]
            aa_bboxes = np.array([h5[part_names[idx] + '_axis_aligned_bbox'][:] for idx in group_idx])
            centers = aa_bboxes[:, :3]
            centers = stabilize(centers, eps=0.03)
            # sort based on the centers
            sorted_idxs = sorted(group_idx, key=lambda idx: tuple(centers[list(group_idx).index(idx)]))
            part_index[group_idx] = np.arange(len(group_idx))
            for i, idx in enumerate(sorted_idxs):
                part_cad_vec_sorted[group_idx[i]] = part_cad_vec[idx]
        part_cad_vec = part_cad_vec_sorted
        
        lens = [len(_) for _ in part_cad_vec]
        
        cad_vec = np.concatenate(part_cad_vec, axis=0)
        
        # quantization for the continuous parameters
        cad_sequence = CADSequence.from_vector(cad_vec, n=self.n_bins)
        bbox_info = compute_bounding_box(obj_path)
        max_point = np.array([bbox_info[0][1], bbox_info[1][1], bbox_info[2][1]])
        min_point = np.array([bbox_info[0][0], bbox_info[1][0], bbox_info[2][0]])
        bbox = np.stack([max_point, min_point], axis=0)  # (max+min)/2 \approx axis_aligned_bbox[:3]
        cad_sequence.bbox = bbox[0] - bbox[1]
        cad_sequence.normalize()

        if self.need_quantization:
            cad_sequence.numericalize(n=self.n_bins)
        cad_vec = cad_sequence.to_vector(MAX_N_EXT, MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)
        
        if cad_vec is None:
            return None
        
        # decouple into part cads
        part_cad_vec = []
        tot = 0
        for i in range(len(lens)):
            if i == 0:
                part_cad_vec.append(cad_vec[: lens[i]])
            else:
                part_cad_vec.append(cad_vec[tot: tot + lens[i]])
            tot += lens[i]

        assert len(part_name_embeddings) == len(part_cad_vec)
        assert len(part_index) == len(part_cad_vec)
        
        return {
            "part_cad": part_cad_vec,
            "base_part_names": base_part_names,
            "part_name_emb": part_name_embeddings,
            "part_index": part_index,
            "img_tex": image_textured,
            "data_id": data_id,
        }



    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':
    np.random.seed(16)
    for cat in ['chair', 'table', 'storagefurniture']:
        out = Path(f'data/trassembler_data/{cat}_pkl')
        if out.exists():
            # remove the folder
            import shutil
            shutil.rmtree(out)
        out.mkdir(exist_ok=True)
        
        train_ids = open(f'data/splits/{cat}_train_ids.txt').read().splitlines()
        train_ids = [int(i) for i in train_ids]
        test_ids = open(f'data/splits/{cat}_test_ids.txt').read().splitlines()
        test_ids = [int(i) for i in test_ids]

        ds = CADDataset(cat=cat, data_ids=train_ids+test_ids)
        ds.dump_data_to_disk(out)


