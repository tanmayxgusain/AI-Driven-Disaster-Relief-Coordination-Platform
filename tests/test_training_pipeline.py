import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import torch
import numpy as np
from PIL import Image

from vision.train import (
    TrainConfig, create_loss_function, train_epoch, save_checkpoint
)
from vision.evaluate import compute_confusion_matrix, calculate_metrics, evaluate_epoch
from vision.unet_model import SiameseUNet
from vision.xbd_loader import XBDDataset
from torch.utils.data import DataLoader

@pytest.fixture
def synthetic_train_dataset(tmp_path):
    # Create dataset structure
    for split in ["train", "val"]:
        images_dir = tmp_path / split / "images"
        labels_dir = tmp_path / split / "labels"
        masks_dir = tmp_path / split / "masks"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        sample_id = "test_001"
        Image.new('RGB', (1024, 1024), color='green').save(images_dir / f"{sample_id}_pre_disaster.png")
        Image.new('RGB', (1024, 1024), color='red').save(images_dir / f"{sample_id}_post_disaster.png")
        
        # Numeric mask
        mask_np = np.zeros((1024, 1024), dtype=np.uint8)
        mask_np[100:200, 100:200] = 3
        Image.fromarray(mask_np).save(masks_dir / f"{sample_id}_post_disaster.png")
        
        wkt_poly = "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"
        annotation = {
            "features": {
                "xy": [{"properties": {"subtype": "major-damage"}, "wkt": wkt_poly}]
            }
        }
        with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
            json.dump(annotation, f)
            
    return tmp_path

def test_loss_function():
    config = TrainConfig(ignore_index=4)
    criterion = create_loss_function(config)
    
    logits = torch.randn(2, 4, 10, 10)
    # Mask containing mostly 4, and some 0-3
    target = torch.full((2, 10, 10), 4, dtype=torch.long)
    target[0, 0, 0] = 0
    target[1, 1, 1] = 2
    
    loss = criterion(logits, target)
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_weighted_loss_function_valid():
    config = TrainConfig(ignore_index=4, class_weights=(0.5, 1.0, 1.5, 2.0), device="cpu")
    criterion = create_loss_function(config)
    
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert torch.allclose(criterion.weight, torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float32))
    assert criterion.ignore_index == 4

def test_weighted_loss_function_invalid_length():
    config = TrainConfig(class_weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="exactly 4 values"):
        create_loss_function(config)

def test_weighted_loss_function_invalid_value():
    invalid_configs = [
        TrainConfig(class_weights=(1.0, -1.0, 1.0, 1.0)),
        TrainConfig(class_weights=(1.0, 1.0, 0.0, 1.0)),
        TrainConfig(class_weights=(1.0, float('nan'), 1.0, 1.0)),
        TrainConfig(class_weights=(1.0, float('inf'), 1.0, 1.0))
    ]
    for config in invalid_configs:
        with pytest.raises(ValueError, match="finite and > 0"):
            create_loss_function(config)

def test_metrics():
    # Synthetic preds and targets
    # Classes 0-3, Ignore 4
    preds = torch.tensor([
        [[0, 1, 2], [3, 0, 0]]
    ])
    targets = torch.tensor([
        [[0, 1, 4], [4, 0, 0]]
    ])
    
    conf_matrix = compute_confusion_matrix(preds, targets, num_classes=4, ignore_index=4)
    metrics = calculate_metrics(conf_matrix)
    
    # 4 valid pixels: (0,0), (1,1), (0,0), (0,0)
    # They all match exactly (100% accuracy)
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["mean_iou"] == pytest.approx(1.0, rel=1e-5)
    assert metrics["mean_dice"] == pytest.approx(1.0, rel=1e-5)

def test_dataloader(synthetic_train_dataset):
    dataset = XBDDataset(synthetic_train_dataset, split="train")
    loader = DataLoader(dataset, batch_size=1)
    
    batch = next(iter(loader))
    assert "pre_image" in batch
    assert batch["pre_image"].shape == (1, 3, 256, 256)

def test_forward_loss_backward():
    model = SiameseUNet()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    target = torch.randint(0, 5, (1, 256, 256), dtype=torch.long)
    
    logits = model(pre, post)
    loss = criterion(logits, target)
    loss.backward()
    
    assert model.enc1.conv[0].weight.grad is not None
    assert torch.isfinite(loss)

def test_training_epoch(synthetic_train_dataset):
    dataset = XBDDataset(synthetic_train_dataset, split="train")
    loader = DataLoader(dataset, batch_size=1)
    
    model = SiameseUNet()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss = train_epoch(model, loader, criterion, optimizer, device="cpu")
    assert isinstance(loss, float)
    assert loss > 0

def test_validation_epoch(synthetic_train_dataset):
    dataset = XBDDataset(synthetic_train_dataset, split="val")
    loader = DataLoader(dataset, batch_size=1)
    
    model = SiameseUNet()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    
    metrics = evaluate_epoch(model, loader, criterion, device="cpu", num_classes=4, ignore_index=4)
    
    assert "loss" in metrics
    assert "mean_iou" in metrics
    assert isinstance(metrics["loss"], float)
    
    # Check no gradients were accumulated
    assert model.enc1.conv[0].weight.grad is None

def test_checkpointing(tmp_path):
    model = SiameseUNet()
    optimizer = torch.optim.Adam(model.parameters())
    
    config = TrainConfig(checkpoint_dir=str(tmp_path), class_weights=(0.27, 1.0, 1.55, 1.19))
    val_metrics = {"loss": 0.5, "mean_iou": 0.8}
    
    save_checkpoint(model, optimizer, None, 1, 0.6, val_metrics, config, "test.pt")
    
    checkpoint_path = tmp_path / "test.pt"
    assert checkpoint_path.exists()
    
    # Load and verify
    ckpt = torch.load(checkpoint_path, weights_only=False)
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == 0.5
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert ckpt["config"]["class_weights"] == (0.27, 1.0, 1.55, 1.19)

def test_cpu_smoke_run(synthetic_train_dataset, tmp_path):
    from vision.train import train_pipeline
    
    config = TrainConfig(
        dataset_root=str(synthetic_train_dataset),
        train_split="train",
        val_split="val",
        max_train_samples=16,
        max_val_samples=8,
        batch_size=1,
        epochs=1,
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints")
    )
    
    model = train_pipeline(config)
    
    assert os.path.exists(tmp_path / "checkpoints" / "last.pt")
    assert os.path.exists(tmp_path / "checkpoints" / "best.pt")
    
    # Check resume logic runs
    model_resume = train_pipeline(config, resume_checkpoint=str(tmp_path / "checkpoints" / "last.pt"))
    assert model_resume is not None
