"""
Preprocessing logic for Computer Vision pipeline.
Supports paired pre/post image transformations with matching mask augmentation.
"""

import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np

def get_train_transform(pre_image, post_image, mask):
    """
    Applies consistent spatial augmentations to pre/post images and the damage mask.
    All three inputs receive the exact same geometric transforms.
    
    Expected inputs:
        pre_image (PIL Image): RGB image
        post_image (PIL Image): RGB image
        mask (PIL Image): 2D mask with class indices
        
    Outputs:
        pre_tensor (torch.Tensor): shape (3, 256, 256), float32, ImageNet normalized
        post_tensor (torch.Tensor): shape (3, 256, 256), float32, ImageNet normalized
        mask_tensor (torch.Tensor): shape (256, 256), int64 (long), NOT normalized
    """
    # Resize to 256x256
    # Images use default bilinear interpolation
    pre_image = TF.resize(pre_image, [256, 256])
    post_image = TF.resize(post_image, [256, 256])
    # Mask uses nearest neighbor to avoid blurring discrete class labels
    mask = TF.resize(mask, [256, 256], interpolation=InterpolationMode.NEAREST)

    # Random Horizontal Flip (50% chance)
    if random.random() > 0.5:
        pre_image = TF.hflip(pre_image)
        post_image = TF.hflip(post_image)
        mask = TF.hflip(mask)

    # Random Vertical Flip (50% chance)
    if random.random() > 0.5:
        pre_image = TF.vflip(pre_image)
        post_image = TF.vflip(post_image)
        mask = TF.vflip(mask)

    # Random Affine (Rotation + Translation + Scale)
    angle = random.uniform(-15, 15)
    translate_x = int(random.uniform(-0.1, 0.1) * 256)
    translate_y = int(random.uniform(-0.1, 0.1) * 256)
    scale = random.uniform(0.9, 1.1)
    
    pre_image = TF.affine(pre_image, angle, (translate_x, translate_y), scale, 0)
    post_image = TF.affine(post_image, angle, (translate_x, translate_y), scale, 0)
    # Mask affine also needs NEAREST interpolation
    mask = TF.affine(mask, angle, (translate_x, translate_y), scale, 0, interpolation=InterpolationMode.NEAREST)

    # Convert to Tensors
    pre_tensor = TF.to_tensor(pre_image)
    post_tensor = TF.to_tensor(post_image)
    mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

    # ImageNet Normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pre_tensor = TF.normalize(pre_tensor, mean=mean, std=std)
    post_tensor = TF.normalize(post_tensor, mean=mean, std=std)

    return pre_tensor, post_tensor, mask_tensor

def get_val_transform(pre_image, post_image, mask):
    """
    Validation preprocessing (no random augmentations, just resize and normalize).
    """
    pre_image = TF.resize(pre_image, [256, 256])
    post_image = TF.resize(post_image, [256, 256])
    mask = TF.resize(mask, [256, 256], interpolation=InterpolationMode.NEAREST)

    pre_tensor = TF.to_tensor(pre_image)
    post_tensor = TF.to_tensor(post_image)
    mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pre_tensor = TF.normalize(pre_tensor, mean=mean, std=std)
    post_tensor = TF.normalize(post_tensor, mean=mean, std=std)

    return pre_tensor, post_tensor, mask_tensor
