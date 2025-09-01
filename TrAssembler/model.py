import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
from copy import deepcopy
import h5py
import os
import hydra
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../GMFlow')))
from GMFlow.lib.models.diffusions.gmflow import GMFlow
from utils.misc import render_nvdiffrast_rgb
from cadlib.macro import *
from einops import rearrange
from dataset import DiskCADDataset, collate_fn
from cadlib.visualize import vec2CADsolid
from utils.io import cad_to_obj
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from utils.transformer_utils import TransformerDecoder
from torch_scatter import scatter_add
from flow_matching.path import CondOTProbPath
from flow_matching.utils import ModelWrapper
import json
import nvdiffrast.torch as dr

np.set_printoptions(precision=2, suppress=True)

T_SCALE = 1000
EPS = 1e-4
TRANS_RATIO = 0.7

class WrappedModel(ModelWrapper):
    def forward(self, x, t, **batch):
        x = rearrange(x, 'B N P L -> B P L N')
        x[batch['args_mask'] == 0] = 0  # also mask input
        u_gm = self.model(x, t, **batch)
        # u_gm['means'][(batch['args_mask'] == 0)[..., None, :].expand(-1, -1, -1, self.model.num_g, N_ARGS)] = 0
        # u_gm['logweights'][(batch['command_mask'] == 0)[..., None, None].expand(-1, -1, -1, self.model.num_g, -1)] = 0
        u_gm['means'] = rearrange(u_gm['means'], 'B P L K N -> B K N P L')
        u_gm['logweights'] = rearrange(u_gm['logweights'], 'B P L K 1 -> B K 1 P L')
        return u_gm


def sinosoidal_positional_encoding(position, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=position.device) * -emb)
    emb = position.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad if needed
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (position.shape[0], embedding_dim)
    return emb


def sample_forward_transition(x_t_low, t_low, t_high, noise):
    bs = x_t_low.size(0)
    if t_low.dim() == 0:
        t_low = t_low.expand(bs)
    if t_high.dim() == 0:
        t_high = t_high.expand(bs)

    std_low = t_low.reshape(*t_low.size(), *((x_t_low.dim() - t_low.dim()) * [1]))
    mean_low = 1 - std_low
    std_high = t_high.reshape(*t_high.size(), *((x_t_low.dim() - t_high.dim()) * [1]))
    mean_high = 1 - std_high

    mean_trans = mean_high / mean_low
    std_trans = (std_high ** 2 - (mean_trans * std_low) ** 2).sqrt()
    return x_t_low * mean_trans + noise * std_trans


def reverse_transition(denoising_output, x_t_high, t_low, t_high, eps=1e-6, prediction_type='u'):
    if isinstance(denoising_output, dict):
        x_t_high = x_t_high.unsqueeze(-2)

    bs = x_t_high.size(0)
    if not isinstance(t_low, torch.Tensor):
        t_low = torch.tensor(t_low, device=x_t_high.device)
    if not isinstance(t_high, torch.Tensor):
        t_high = torch.tensor(t_high, device=x_t_high.device)
    if t_low.dim() == 0:
        t_low = t_low.expand(bs)
    if t_high.dim() == 0:
        t_high = t_high.expand(bs)
    t_low = t_low.reshape(*t_low.size(), *((x_t_high.dim() - t_low.dim()) * [1]))
    t_high = t_high.reshape(*t_high.size(), *((x_t_high.dim() - t_high.dim()) * [1]))

    sigma = t_high
    sigma_to = t_low
    alpha = 1 - sigma
    alpha_to = 1 - sigma_to

    sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
    alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
    beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2

    c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
    c2 = beta_over_sigma_sq * alpha_to

    if isinstance(denoising_output, dict):
        c3 = beta_over_sigma_sq * sigma_to ** 2
        if prediction_type == 'u':
            means_x_0 = x_t_high - sigma * denoising_output['means']
            logstds_x_t_low = torch.logaddexp(
                (denoising_output['logstds'] + torch.log((sigma * c2).clamp(min=eps))) * 2,
                torch.log(c3.clamp(min=eps))
            ) / 2
        elif prediction_type == 'x0':
            means_x_0 = denoising_output['means']
            logstds_x_t_low = torch.logaddexp(
                (denoising_output['logstds'] + torch.log(c2.clamp(min=eps))) * 2,
                torch.log(c3.clamp(min=eps))
            ) / 2
        else:
            raise ValueError('Invalid prediction_type.')
        means_x_t_low = c1 * x_t_high + c2 * means_x_0
        return dict(
            means=means_x_t_low,
            logstds=logstds_x_t_low,
            logweights=denoising_output['logweights'])

    else:  # sample mode
        c3_sqrt = beta_over_sigma_sq ** 0.5 * sigma_to
        noise = torch.randn_like(denoising_output)
        if prediction_type == 'u':
            x_0 = x_t_high - sigma * denoising_output
        elif prediction_type == 'x0':
            x_0 = denoising_output
        else:
            raise ValueError('Invalid prediction_type.')
        x_t_low = c1 * x_t_high + c2 * x_0 + c3_sqrt * noise
        return x_t_low


def gm_kl_loss(gm, sample, mask, eps=1e-4):
    """
    Gaussian mixture KL divergence loss (without constant terms), a.k.a. GM NLL loss.

    Args:
        gm (dict):
            means (torch.Tensor): (bs, *, num_gaussians, N)
            logstds (torch.Tensor): (bs, *, 1, 1)
            logweights (torch.Tensor): (bs, *, num_gaussians, 1)
        sample (torch.Tensor): (bs, *, N)
        mask (torch.Tensor): (bs, *, N)

    Returns:
        torch.Tensor: (bs, *)
    """
    means = gm['means']
    logstds = gm['logstds']
    logweights = gm['logweights']

    inverse_stds = torch.exp(-logstds).clamp(max=1 / eps)
    diff_weighted = (sample.unsqueeze(-2) - means) * inverse_stds  # (bs, *, num_gaussians, N)
    gaussian_ll = ((-0.5 * diff_weighted.square() - logstds) * mask.unsqueeze(-2)).sum(dim=-1)
    gm_nll = -torch.logsumexp(gaussian_ll + logweights.squeeze(-1), dim=-1)
    return gm_nll


# Data module remains essentially unchanged.
class CADDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, cat, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        train_ids = open(f'data/splits/{cat}_train_ids.txt').read().splitlines()
        train_ids = [int(i) for i in train_ids]
        test_ids = open(f'data/splits/{cat}_test_ids.txt').read().splitlines()
        test_ids = [int(i) for i in test_ids]

        self.train_ds = DiskCADDataset(data_dir, white_list=train_ids)
        self.val_ds = DiskCADDataset(data_dir, white_list=test_ids)

        print(f"Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")

        # repeat train_ds a few times to augment the training data
        self.train_ds = torch.utils.data.ConcatDataset([deepcopy(self.train_ds) for _ in range(5)])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=10,
                          collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=1,
                          collate_fn=collate_fn, shuffle=True)

class GMFlowModel(pl.LightningModule):
    """TrAssembler model for predicting continuous CAD parameters.
    
    This model uses a Gaussian Mixture Flow approach to predict continuous
    parameters for CAD commands given discrete command structures and input images.
    
    Args:
        args: Configuration object containing model parameters
        embed_dim: Embedding dimension for transformers
        num_heads: Number of attention heads
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
        scaling_factor: Scaling factor for parameters
        args_range: Range of argument values
        shift: Time shift parameter for diffusion
    """
    def __init__(self, args, embed_dim, num_heads, dropout, bias, scaling_factor, args_range, shift=1.0):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim

        self.shift = shift

        # Command embedding (same as before)
        self.command_emb = nn.Embedding(num_embeddings=len(ALL_COMMANDS), embedding_dim=embed_dim, padding_idx=EOS_IDX)

        # For each command type, set up an MLP to embed the corresponding arguments.
        self.args_emb = nn.ModuleList()
        for i in range(len(ALL_COMMANDS)):
            n_args = CMD_ARGS_MASK[i].sum()
            self.args_emb.append(nn.Sequential(
                nn.Linear(n_args, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            ))

        self.register_buffer('cmd_args_mask', (torch.from_numpy(CMD_ARGS_MASK) > 0).bool())
        self.register_buffer('args_range', torch.from_numpy(args_range))
        self.cmd_args_idx = [
            torch.from_numpy(np.where(CMD_ARGS_MASK[i] > 0)[0])
            for i in range(len(ALL_COMMANDS))
        ]

        self.t_emb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # For image processing we still use dinov2
        dino_size = 'b'
        if dino_size == 's':
            dino_dim = 384
        elif dino_size == 'b':
            dino_dim = 768
        elif dino_size == 'l':
            dino_dim = 1024
        else:
            raise ValueError("Invalid dino size")

        # Transformer decoders (unchanged structure)
        self.sketch_transformer = TransformerDecoder(
            depth=6,
            heads=8,
            mlp_dim=embed_dim,
            dim_head=64,
            dropout=dropout,
            emb_dropout=0.,
            norm='layer',
            context_dim=dino_dim,
            num_tokens=256,
            token_dim=3 * embed_dim,
            dim=embed_dim,
            add_pos_embedding=False,
        )

        self.part_transformer = TransformerDecoder(
            depth=6,
            heads=8,
            mlp_dim=embed_dim,
            dim_head=64,
            dropout=dropout,
            emb_dropout=0.,
            norm='layer',
            context_dim=dino_dim,
            num_tokens=256,
            token_dim=3 * embed_dim,
            dim=embed_dim,
            add_pos_embedding=False,
        )

        self.decoder_head = nn.ModuleList()
        for i in range(len(ALL_COMMANDS)):
            n_args = CMD_ARGS_MASK[i].sum()
            self.decoder_head.append(TransformerDecoder(
                depth=6,
                heads=8,
                mlp_dim=embed_dim,
                dim_head=64,
                dropout=dropout,
                emb_dropout=0.,
                norm='layer',
                context_dim=4 * embed_dim,
                num_tokens=args.decoder_num_tokens,
                token_dim=1,
                dim=embed_dim)
            )
        
        self.num_g = 32
        # For diffusion we work in continuous regression mode so the output is a single value (noise)
        self.outs = nn.ModuleList()
        for i in range(len(ALL_COMMANDS)):
            n_args = CMD_ARGS_MASK[i].sum()
            self.outs.append(nn.Linear(n_args * embed_dim, self.num_g * (n_args + 1)))

        # Load dinov2 (frozen)
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{dino_size}14')
        for param in self.dinov2.parameters():
            param.requires_grad = False

        self.img_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[-1., -1., -1.], std=[2.0, 2.0, 2.0]),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.random_affine = transforms.RandomAffine(degrees=0, translate=(0.1, 0.2), scale=(0.8, 1.2))

        self.part_name_encoder = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # New time embedding for diffusion conditioning.
        # We project a sinusoidal embedding (of size embed_dim) to a vector of size 3*embed_dim,
        # which is then added to the line embedding (which has size 3*embed_dim).
        self.diffusion_time_embed = nn.Sequential(
            nn.Linear(embed_dim, 3 * embed_dim),
            nn.ReLU(),
            nn.Linear(3 * embed_dim, 3 * embed_dim)
        )
        self.path = CondOTProbPath()
        
        self.logstd_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(3 * embed_dim, 3 * embed_dim),
            nn.SiLU(),
            nn.Linear(3 * embed_dim, 3 * embed_dim),
            nn.SiLU(),
            nn.Linear(3 * embed_dim, 1),
        )

        # load mean scale
        stats = json.load(open(f'data/{args.category}_mean_scale_stats.json'))
        self.mean_scale = {}
        for k, v in stats.items():
            self.mean_scale[int(k)] = {
                'mean': torch.tensor(v['mean'], dtype=torch.float32),
                'scale': torch.tensor(v['std'], dtype=torch.float32),
            }
            
        self.ctx = dr.RasterizeCudaContext()
        
        
        
    def normalize(self, args, args_mask):
        for i in range(args.shape[-1]):
            if i in self.mean_scale:
                b_idx, p_idx, l_idx = torch.nonzero(args_mask[..., i] > 0, as_tuple=True)
                args[b_idx, p_idx, l_idx, i] -= self.mean_scale[i]['mean']
                args[b_idx, p_idx, l_idx, i] /= self.mean_scale[i]['scale']
                
    def denormalize(self, args, args_mask):
        for i in range(args.shape[-1]):
            if i in self.mean_scale:
                b_idx, p_idx, l_idx = torch.nonzero(args_mask[..., i] > 0, as_tuple=True)
                args[b_idx, p_idx, l_idx, i] *= self.mean_scale[i]['scale']
                args[b_idx, p_idx, l_idx, i] += self.mean_scale[i]['mean']

    def forward(self, x, t, **batch):
        """
        Forward pass of the TrAssembler model.
        
        Args:
            x: Noisy CAD arguments tensor of shape [B, P, L, N]
            t: Diffusion timestep tensor of shape [B]
            **batch: Batch dictionary containing:
                - command: Command type tensor [B, P, L]
                - command_mask: Command validity mask [B, P, L]
                - part_name_embedding: Part semantic embeddings [B, P, D]
                - part_index: Part ordering indices [B, P]
                - img_tex: Input image tensor [B, C, H, W]
                
        Returns:
            Dictionary containing Gaussian mixture parameters:
                - means: Mixture component means [B, P, L, K, N]
                - logstds: Log standard deviations [B, P, L, K, 1]
                - logweights: Log mixture weights [B, P, L, K, 1]
        """
        command = batch['command']
        command_mask = batch['command_mask']
        # args = batch['args']  # now x_t: the noisy version of the ground truth
        args = x
        part_name_emb = batch['part_name_embedding']
        part_index = batch['part_index']
        img = batch['img_tex']

        self.dinov2.eval()
        with torch.no_grad():
            if self.training and torch.rand(1) > 0.5:
                img = self.random_affine(img)
            img = self.img_transform(img)
            img_tokens = self.dinov2.forward_features(img)['x_norm_patchtokens']  # B x NT x D

        # Process commands and arguments
        n_batch, n_part, n_line = command.shape
        command_emb = self.command_emb(command)  # [B, P, L, D]

        # Compute args embedding per command type
        args_emb_list = []
        for i in range(len(ALL_COMMANDS)):
            mask = command == i
            mask_idx = torch.nonzero(mask)  # K x 3
            mask_idx_1d = mask_idx[:, 0] * n_part * n_line + mask_idx[:, 1] * n_line + mask_idx[:, 2]
            if mask.sum() > 0:
                # Select only the arguments corresponding to active (non-padded) positions
                args_masked = args[mask][:, np.where(CMD_ARGS_MASK[i] > 0)[0]]  # K x NA
                args_emb_masked = self.args_emb[i](args_masked)  # K x D
                args_emb_list.append(scatter_add(args_emb_masked, mask_idx_1d, dim=0, 
                                                 dim_size=n_batch * n_part * n_line).reshape(n_batch, n_part, n_line, -1))
        # Sum over contributions from all command types
        args_emb = torch.stack(args_emb_list, dim=0).sum(dim=0)  # [B, P, L, D]

        # Compute positional embedding for each line token
        line_pos_emb = sinosoidal_positional_encoding(torch.arange(n_line, device=args_emb.device) + 1, self.embed_dim)  # [L, D]
        line_pos_emb = line_pos_emb[None, None].expand(n_batch, n_part, n_line, -1)

        # Concatenate command, argument and positional embeddings -> [B, P, L, 3D]
        line_emb = torch.cat([command_emb, args_emb, line_pos_emb], dim=-1)

        # Condition on the diffusion timestep if provided.
        if t is not None:
            if t.dim() == 0:
                t = t.unsqueeze(0).expand(n_batch)  # Ensure t is of shape [B]
            # t is of shape [B]; get sinusoidal encoding and project to 3*embed_dim.
            t_embed = self.diffusion_time_embed(sinosoidal_positional_encoding(t, self.embed_dim))
            # Add time embedding to each line token.
            line_emb = line_emb + t_embed[:, None, None, :]

        # Zero-out padded lines.
        line_emb = line_emb * command_mask.unsqueeze(-1)
        line_emb = rearrange(line_emb, 'B P L D -> (B P) L D')

        line_mask = rearrange(command_mask, 'B P L -> (B P) L')
        line_emb_out = self.sketch_transformer(line_emb, 
                                               context=img_tokens[:, None].repeat(1, n_part, 1, 1).reshape(-1, *img_tokens.shape[1:]),
                                               x_mask=line_mask)
        line_emb_out = line_emb_out * line_mask.unsqueeze(-1)
        line_emb_out = rearrange(line_emb_out, '(B P) L D -> B P L D', B=n_batch)
        part_emb = line_emb_out.sum(dim=-2)  # aggregate along lines

        # Part positional encoding
        part_pos_emb = sinosoidal_positional_encoding(part_index.reshape(-1) + 1, self.embed_dim).view(*part_index.shape, -1)
        part_name_emb = self.part_name_encoder(part_name_emb)
        part_emb = torch.cat([part_emb, part_name_emb, part_pos_emb], dim=-1)
        part_emb_mask = command_mask.any(dim=2).float()
        part_emb = part_emb * part_emb_mask.unsqueeze(-1)

        part_emb_out = self.part_transformer(part_emb, context=img_tokens, x_mask=part_emb_mask)
        part_emb_out = part_emb_out * part_emb_mask.unsqueeze(-1)

        # Concatenate part embedding to each line embedding for decoding.
        part_emb_out = part_emb_out.unsqueeze(-2).repeat(1, 1, n_line, 1)
        line_emb_dec = torch.cat([line_emb_out, part_emb_out], dim=-1)
        line_emb_dec = rearrange(line_emb_dec, 'B P L D -> (B P) L D')
        line_emb_dec = line_emb_dec * line_mask.unsqueeze(-1)

        context = torch.cat([
            rearrange(part_name_emb, 'B P D -> B P 1 D').repeat(1, 1, n_line, 1),
            part_emb_out,
            rearrange(line_emb_dec, '(B P) L D -> B P L D', B=n_batch)
        ], dim=-1)
        context = rearrange(context, 'B P L D -> (B P) L D')
        context = context * line_mask.unsqueeze(-1)

        # Decode noise prediction for each command type.
        args_dec_list = []
        for i in range(len(ALL_COMMANDS)):
            n_args = CMD_ARGS_MASK[i].sum()
            mask = command == i
            mask_idx = torch.nonzero(mask)
            mask_idx_1d = mask_idx[:, 0] * n_part * n_line + mask_idx[:, 1] * n_line + mask_idx[:, 2]
            if mask.sum() > 0:
                x_mask = line_mask[:, :, None].repeat(1, 1, n_args).reshape(n_batch * n_part, -1)
                token = torch.zeros(n_batch * n_part, n_line * n_args, 1, device=context.device)
                decoder_out = self.decoder_head[i](token, x_mask=x_mask, context_mask=line_mask, context=context)  # (B * P, L * N, D)
                decoder_out = rearrange(decoder_out, '(B P) (L N) D -> B P L (N D)', P=n_part, L=n_line)
                token_out = self.outs[i](decoder_out)  # (B, P, L, K * (N + 1))
                token_out = rearrange(token_out, 'B P L (K D) -> B P L K D', K=self.num_g)  # (B, P, L, K, N + 1)
                out = scatter_add(token_out[mask], mask_idx_1d, dim=0, dim_size=n_batch * n_part * n_line)  # (B * P * L, K, N + 1)
                out = torch.cat([scatter_add(out[..., :n_args], self.cmd_args_idx[i].to(out.device), dim=2, dim_size=N_ARGS), out[..., n_args:]], dim=2)
                args_dec_list.append(out.reshape(n_batch, n_part, n_line, self.num_g, -1))
        # Sum contributions from each command type to obtain final noise prediction.
        args_dec = torch.stack(args_dec_list, dim=0).sum(dim=0)  # (B, P, L, K, N + 1)
        
        # (B, P, L, K, N), (B, P, L, K, 1)
        means, logweights = args_dec.split([N_ARGS, 1], dim=-1)
        logweights = logweights.log_softmax(dim=-2)
        logstds = self.logstd_mlp(t_embed.detach()).reshape(n_batch, 1, 1, 1, 1)
        return {'means': means,
                'logstds': logstds,
                'logweights': logweights}

    def compute_loss(self, pred_args, gt_args, args_mask):
        loss = torch.sum(args_mask.float() * (pred_args - gt_args).abs().pow(2)) / (args_mask.sum() + 1e-6)
        return loss
    
    def training_step(self, batch, batch_idx):
        x_0 = batch['args']  # ground-truth continuous args (x₀)
        x_0[batch['args_mask'] == 0] = 0
        
        # normalize gt args
        self.normalize(x_0, batch['args_mask'])
        
        B = x_0.shape[0]
        # Sample a random diffusion timestep for each sample
        t = torch.rand((B,), device=self.device)
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)
        t = t.clamp(min=EPS)

        t_low = t * (1 - TRANS_RATIO)
        t_low = torch.minimum(t_low, t - EPS).clamp(min=0)

        noise_0 = torch.randn_like(x_0)
        noise_1 = torch.randn_like(x_0)

        sigma_t_low = t_low
        alpha_t_low = 1 - sigma_t_low

        x_t_low = alpha_t_low.reshape(B, 1, 1, 1) * x_0 + sigma_t_low.reshape(B, 1, 1, 1) * noise_0
        x_t = sample_forward_transition(x_t_low, t_low, t, noise_1)

        x_t[batch['args_mask'] == 0] = 0

        u_gm = self(x_t, t * T_SCALE, **batch)
        x_t_low_gm = reverse_transition(u_gm, x_t, t_low, t)

        loss = gm_kl_loss(x_t_low_gm, x_t_low, mask=batch['args_mask'])
        loss = (loss * batch['command_mask']).sum() / (batch['command_mask'].sum() + 1e-6)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        # Save original args before modifying
        orig_args = batch['args'].clone()
        
        x_0 = batch['args']  # ground-truth continuous args (x₀)
        x_0[batch['args_mask'] == 0] = 0
        
        # normalize gt args
        self.normalize(x_0, batch['args_mask'])
        
        B = x_0.shape[0]
        # Sample a random diffusion timestep for each sample 
        t = torch.rand((B,), device=self.device)
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)
        t = t.clamp(min=EPS)

        t_low = t * (1 - TRANS_RATIO)
        t_low = torch.minimum(t_low, t - EPS).clamp(min=0)

        noise_0 = torch.randn_like(x_0)
        noise_1 = torch.randn_like(x_0)

        sigma_t_low = t_low
        alpha_t_low = 1 - sigma_t_low

        x_t_low = alpha_t_low.reshape(B, 1, 1, 1) * x_0 + sigma_t_low.reshape(B, 1, 1, 1) * noise_0
        x_t = sample_forward_transition(x_t_low, t_low, t, noise_1)

        x_t[batch['args_mask'] == 0] = 0

        u_gm = self(x_t, t * T_SCALE, **batch)
        x_t_low_gm = reverse_transition(u_gm, x_t, t_low, t)

        loss = gm_kl_loss(x_t_low_gm, x_t_low, mask=batch['args_mask'])
        loss = (loss * batch['command_mask']).sum() / (batch['command_mask'].sum() + 1e-6)
        
        # Restore original args
        batch['args'] = orig_args
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        
        pred_args = self.sample(batch)
        
        # denormalize pred args and gt args
        self.denormalize(pred_args, batch['args_mask'])
        
        # (Below is similar to your original inference code.)
        for i in range(len(batch['args'])):
            data_id = batch['data_id'][i]
            # Load ground truth
            data_path = os.path.join(f'data/raw_annotated/{self.args.category}', str(data_id), "cad.h5")
            h5 = h5py.File(data_path, 'r')
            part_names = np.array([n for n in h5 if n != 'vec' and '_bbox' not in n])
            part_cad_vec = [h5[n][:] for n in part_names]
            cad_vec_gt = np.concatenate(part_cad_vec, axis=0)
            try:
                full_shape = vec2CADsolid(cad_vec_gt, is_numerical=False)
                cad_to_obj(full_shape, 'cad.obj')
                gt_img = render_nvdiffrast_rgb('cad.obj', ctx=self.ctx)
            except Exception as e:
                print(e)
                break

            # Build predicted CAD vector from the generated args
            cad_vec = []
            cmd = batch['command'][i]  # [P, L]
            cmd_mask = batch['command_mask'][i]  # [P, L]
            arg_mask = batch['args_mask'][i]  # [P, L, N]
            args = batch['args'][i]  # original args (will be overwritten)
            # Overwrite args with predicted ones from diffusion sampling
            args[arg_mask > 0] = pred_args[i][arg_mask > 0]
            for p in range(cmd_mask.shape[0]):
                if cmd_mask[p][0] > 0:  # valid part
                    for l in range(cmd_mask.shape[1]):
                        if cmd_mask[p][l] > 0:
                            arg = args[p][l].cpu().numpy()
                            cmd_type = cmd[p][l].item()
                            if cmd_type == 1:  # Arc command
                                arg[2] /= np.pi / 180.  # Convert angle
                            cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
            cad_vec = np.array(cad_vec)
            try:
                full_shape = vec2CADsolid(cad_vec, is_numerical=False)
                cad_to_obj(full_shape, 'cad.obj')
                pred_img = render_nvdiffrast_rgb('cad.obj', ctx=self.ctx)
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key=f"{batch_idx}", images=[np.concatenate([gt_img, pred_img], axis=1)], step=self.global_step)
                else:
                    self.logger.experiment.add_image(f"{batch_idx}", np.concatenate([gt_img, pred_img], axis=1), dataformats='HWC')
            except Exception as e:
                print(e)
            break

    @torch.no_grad()
    def sample(self, batch, step_callback=None):
        """
        Generate CAD parameters via iterative reverse diffusion.
        
        Args:
            batch: Input batch containing conditioning information
            step_callback: Optional callback function for each denoising step
            
        Returns:
            Generated CAD parameters tensor [B, P, L, N]
        """
        shape = batch['args'].shape
        x_init = torch.randn(shape, device=self.device)
        x_init[batch['args_mask'] == 0] = 0
        
        model = GMFlow(
            denoising=WrappedModel(self),
            num_timesteps=T_SCALE,
            # test_cfg=dict(  # use 2nd-order GM-SDE solver
            #     output_mode='sample',
            #     sampler='GMFlowSDE',
            #     num_timesteps=16,
            #     order=2)
            test_cfg=dict(
                output_mode='mean',
                sampler='FlowEulerODE',
                num_timesteps=32,
                num_substeps=4,
                gm2_correction_steps=2,
                order=2,
                sampler_kwargs=dict(shift=self.shift)),
        ).eval()
        
        x_init = rearrange(x_init, 'B P L N -> B N P L')
        samples = model.forward_test(noise=x_init, step_callback=step_callback, **batch)
        samples = rearrange(samples, 'B N P L -> B P L N')
        samples[batch['args_mask'] == 0] = 0
        return samples

    @torch.no_grad()
    def inference(self, batch, data_root="data/raw_annotated"):
        """
        Use the reverse diffusion process to generate args and then render the corresponding CAD.
        
        Args:
            batch: Input batch containing commands, masks, and other data
            data_root: Root directory for raw annotated data
        
        Returns:
            pred_args: Predicted arguments
            cad_vec: Generated CAD vector
        """
        # Generate args via iterative denoising
        pred_args = self.sample(batch)
        
        self.denormalize(pred_args, batch['args_mask'])
        
        for i in range(len(batch['args'])):
            data_id = batch['data_id'][i]
            
            # Build predicted CAD vector from the generated args
            cad_vec = []
            cmd = batch['command'][i]  # [P, L]
            cmd_mask = batch['command_mask'][i]  # [P, L]
            arg_mask = batch['args_mask'][i]  # [P, L, N]
            args = batch['args'][i]  # original args (will be overwritten)
            
            # Overwrite args with predicted ones from diffusion sampling
            args[arg_mask > 0] = pred_args[i][arg_mask > 0]
            
            for p in range(cmd_mask.shape[0]):
                if cmd_mask[p][0] > 0:  # valid part
                    for l in range(cmd_mask.shape[1]):
                        if cmd_mask[p][l] > 0:
                            arg = args[p][l].cpu().numpy()
                            cmd_type = cmd[p][l].item()
                            
                            if cmd_type == 1:  # Arc command
                                arg[2] /= np.pi / 180.  # Convert angle
                                
                            cad_vec.append(np.concatenate([np.array([cmd_type]), arg]))
            
            cad_vec = np.array(cad_vec)
            return pred_args, cad_vec
        
        return pred_args, None

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
