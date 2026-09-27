import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn
from vision.unet_model import SiameseUNet

def test_model_construction():
    """Test 1: Model construction must succeed."""
    model = SiameseUNet()
    assert isinstance(model, SiameseUNet)

def test_forward_pass_and_shape():
    """Test 2: Verify input [B, 3, 256, 256] -> output [B, 4, 256, 256]"""
    model = SiameseUNet()
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    
    logits = model(pre, post)
    assert logits.shape == (2, 4, 256, 256)

def test_single_batch():
    """Test 3: Verify batch size 1 works."""
    model = SiameseUNet()
    pre = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    
    logits = model(pre, post)
    assert logits.shape == (1, 4, 256, 256)

def test_output_dtype():
    """Test 4: Verify logits are floating point."""
    model = SiameseUNet()
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    
    logits = model(pre, post)
    assert logits.dtype == torch.float32

def test_gradient_flow():
    """Test 5: Run a small synthetic loss and verify gradients exist."""
    model = SiameseUNet()
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    
    logits = model(pre, post)
    loss = logits.mean()
    loss.backward()
    
    # Check if gradients are populated in the first encoder block
    assert model.enc1.conv[0].weight.grad is not None
    assert torch.sum(torch.abs(model.enc1.conv[0].weight.grad)) > 0

def test_shape_mismatch():
    """Test 6: Verify incompatible pre/post dimensions generate a clear error."""
    model = SiameseUNet()
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 128, 128)
    
    with pytest.raises(ValueError, match="Pre and post images must have the same shape"):
        model(pre, post)

def test_crossentropy_compatibility():
    """
    Test 7: Verify that torch.nn.CrossEntropyLoss(ignore_index=4) can consume 
    logits [B,4,H,W] and target [B,H,W] without shape/type errors.
    """
    model = SiameseUNet()
    pre = torch.randn(2, 3, 256, 256)
    post = torch.randn(2, 3, 256, 256)
    
    # Target contains classes 0-3 and background class 4
    target = torch.randint(0, 5, (2, 256, 256), dtype=torch.long)
    
    logits = model(pre, post)
    
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    loss = criterion(logits, target)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)

def test_parameter_count(capsys):
    """Parameter-count test: report the number of trainable parameters."""
    model = SiameseUNet()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # For our base_channels=32 model, it should be reasonable (e.g. ~10M-15M)
    # Just asserting it's greater than 0 and less than 50M to ensure it's not exploding
    print(f"Total trainable parameters: {param_count:,}")
    assert 1_000_000 < param_count < 50_000_000
