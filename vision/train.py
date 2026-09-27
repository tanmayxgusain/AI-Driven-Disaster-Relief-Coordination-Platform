"""
Training pipeline for Siamese U-Net.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass

from .xbd_loader import XBDDataset
from .unet_model import SiameseUNet
from .evaluate import evaluate_epoch

@dataclass
class TrainConfig:
    dataset_root: str = "data/xbd"
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    num_classes: int = 4
    ignore_index: int = 4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def create_dataloaders(config: TrainConfig):
    """Creates train and validation DataLoaders."""
    train_dataset = XBDDataset(config.dataset_root, split="train", is_train=True)
    val_dataset = XBDDataset(config.dataset_root, split="val", is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda")
    )
    
    return train_loader, val_loader

def create_loss_function(config: TrainConfig):
    return nn.CrossEntropyLoss(ignore_index=config.ignore_index)

def create_optimizer_and_scheduler(model: nn.Module, config: TrainConfig):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return optimizer, scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, config, filename):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    filepath = os.path.join(config.checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_metrics.get("loss"),
        'val_metrics': val_metrics,
        'config': config.__dict__
    }
    torch.save(checkpoint, filepath)
    
def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["damage_mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        logits = model(pre_img, post_img)
        loss = criterion(logits, mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)

def train_pipeline(config: TrainConfig):
    """
    Main training execution function.
    """
    set_seed(config.seed)
    
    train_loader, val_loader = create_dataloaders(config)
    
    model = SiameseUNet(in_channels=3, num_classes=config.num_classes).to(config.device)
    criterion = create_loss_function(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, config.device, config.num_classes, config.ignore_index)
        
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)
        
        # Save last checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, config, "last_checkpoint.pt")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, config, "best_model.pt")
            
    return model
