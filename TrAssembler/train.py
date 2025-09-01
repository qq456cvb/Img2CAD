#!/usr/bin/env python3
"""
Training script for TrAssembler model.

This script trains the TrAssembler model that predicts continuous parameters
given discrete CAD command structures.
"""

import argparse
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import GMFlowModel, CADDataModule
import numpy as np


def setup_logger(args, experiment_dir):
    """Setup logger based on configuration."""
    # Create logging directory under experiment directory
    log_dir = Path(experiment_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.wandb:
        logger = WandbLogger(
            project='Img2CAD-TrAssembler', 
            name=f'{args.category}_experiment',
            save_dir=str(log_dir)
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(log_dir), 
            name=f'TrAssembler_{args.category}'
        )
    return logger


def setup_callbacks(args, experiment_dir):
    """Setup training callbacks."""
    # Create checkpoint directory under experiment directory
    checkpoint_dir = Path(experiment_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{step}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,
    )
    return [checkpoint_callback]


@hydra.main(config_path='.', config_name='config', version_base=None)
def train(args: DictConfig):
    """Main training function."""
    # Get Hydra output directory and derive experiment directory
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    
    # Extract the experiment directory from Hydra's output dir (remove '/hydra_outputs' suffix)
    experiment_dir = os.path.dirname(hydra_output_dir)
    
    print(f"Hydra output directory: {hydra_output_dir}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Set random seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Setup logger
    logger = setup_logger(args, experiment_dir)
    
    # Setup callbacks
    callbacks = setup_callbacks(args, experiment_dir)
    
    # Setup data module
    data_module = CADDataModule(
        data_dir=f'data/trassembler_data/{args.category}_pkl',
        cat=args.category, 
        batch_size=args.batch_size, 
    )
    
    # Load model from checkpoint if specified
    if hasattr(args, 'ckpt') and args.ckpt is not None:
        print(f"Loading model from {args.ckpt}")
        model = GMFlowModel.load_from_checkpoint(
            args.ckpt,
            args=args,
            embed_dim=args.network.embed_dim,
            num_heads=args.network.num_heads,
            dropout=args.network.dropout,
            bias=True,
            scaling_factor=1.,
            args_range=np.array([-1., 1.]),
            shift=args.gm.shift
        )
    else:
        print("Training from scratch")
        model = GMFlowModel(
            args=args, 
            embed_dim=args.network.embed_dim,
            num_heads=args.network.num_heads, 
            dropout=args.network.dropout,
            bias=True, 
            scaling_factor=1., 
            args_range=np.array([-1., 1.]),
            shift=args.gm.shift
        )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='cuda' if args.use_gpu else 'cpu',
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        precision=args.precision if hasattr(args, 'precision') else 32,
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    print("Training completed!")


if __name__ == '__main__':
    train()
