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
    train_split: str = "tier1"
    val_split: str = "hold"
    max_train_samples: int = None
    max_val_samples: int = None
    label_source: str = "mask"
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    num_classes: int = 4
    ignore_index: int = 4
    early_stopping_patience: int = 4
    class_weights: tuple = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def create_dataloaders(config: TrainConfig):
    """Creates train and validation DataLoaders."""
    train_dataset = XBDDataset(
        config.dataset_root, split=config.train_split, is_train=True, 
        max_samples=config.max_train_samples, label_source=config.label_source
    )
    val_dataset = XBDDataset(
        config.dataset_root, split=config.val_split, is_train=False, 
        max_samples=config.max_val_samples, label_source=config.label_source
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory if config.device == "cuda" else False
    )
    
    return train_loader, val_loader

def create_loss_function(config: TrainConfig):
    if config.class_weights is None:
        return nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        
    weights = config.class_weights
    if len(weights) != 4:
        raise ValueError(f"class_weights must have exactly 4 values, got {len(weights)}")
        
    for w in weights:
        if not (isinstance(w, (int, float)) and np.isfinite(w) and w > 0):
            raise ValueError(f"All class weights must be finite and > 0, got invalid weight: {w}")
            
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=config.device)
    return nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=config.ignore_index)

def create_optimizer_and_scheduler(model: nn.Module, config: TrainConfig):
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
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
    
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Runs one epoch of training, with optional AMP support.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["damage_mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type=device, enabled=True):
                logits = model(pre_img, post_img)
                loss = criterion(logits, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(pre_img, post_img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)

import time

def train_pipeline(config: TrainConfig, resume_checkpoint: str = None):
    """
    Main training execution function.
    """
    set_seed(config.seed)
    
    train_loader, val_loader = create_dataloaders(config)
    
    model = SiameseUNet(in_channels=3, num_classes=config.num_classes).to(config.device)
    criterion = create_loss_function(config)
    
    if config.class_weights is None:
        print("Loss: CrossEntropyLoss")
        print("Class weights: none")
    else:
        print("Loss: Weighted CrossEntropyLoss")
        print(f"Class weights: {list(config.class_weights)}")
        
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    scaler = torch.amp.GradScaler(device=config.device) if config.device == "cuda" else None
    
    start_epoch = 1
    best_metric = -1.0 # using mean_iou for early stopping
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') and scheduler:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt.get('val_metrics', {}).get("mean_iou", 0.0)
    
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()
        
        train_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device, scaler)
        train_time = time.time() - train_start
        
        val_start = time.time()
        val_metrics = evaluate_epoch(model, val_loader, criterion, config.device, config.num_classes, config.ignore_index)
        val_time = time.time() - val_start
        
        total_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch}/{config.epochs} - Train Time: {train_time:.1f}s - Val Time: {val_time:.1f}s - Total: {total_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - Mean IoU: {val_metrics['mean_iou']:.4f}")
        
        val_loss = val_metrics["loss"]
        current_metric = val_metrics["mean_iou"]
        scheduler.step(val_loss)
        
        # Save last checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, config, "last.pt")
        
        # Save best checkpoint
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_metrics, config, "best.pt")
        else:
            epochs_no_improve += 1
            
        if config.epochs > 1 and epochs_no_improve >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break
            
    return model
