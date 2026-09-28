"""
Dataset loader for the xBD dataset.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from shapely import wkt
from shapely.errors import WKTReadingError

from .preprocessing import get_train_transform, get_val_transform

class XBDDataset(Dataset):
    """
    xBD Dataset Loader.
    
    Expected logical structure:
        data/xbd/
            train/
                images/
                labels/
                
    Expects paired pre/post images and post-disaster JSON annotations.
    
    Damage Classes (Project canonical):
        0 = no damage
        1 = minor damage
        2 = major damage
        3 = destroyed
        4 = background (non-building regions)
    """
    def __init__(self, root_dir, split="train", is_train=True, 
                 pre_img_suffix="_pre_disaster.png",
                 post_img_suffix="_post_disaster.png",
                 post_label_suffix="_post_disaster.json",
                 post_mask_suffix="_post_disaster.png",
                 unclassified_policy="ignore",
                 label_source="mask",
                 max_samples=None):
        
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, "images")
        self.labels_dir = os.path.join(self.split_dir, "labels")
        self.masks_dir = os.path.join(self.split_dir, "masks")
        self.is_train = is_train
        
        if label_source not in ["mask", "json"]:
            raise ValueError("label_source must be 'mask' or 'json'")
        self.label_source = label_source
        self.max_samples = max_samples
        
        if unclassified_policy not in ["ignore", "no_damage"]:
            raise ValueError("unclassified_policy must be 'ignore' or 'no_damage'")
        self.unclassified_policy = unclassified_policy
        
        self.pre_img_suffix = pre_img_suffix
        self.post_img_suffix = post_img_suffix
        self.post_label_suffix = post_label_suffix
        self.post_mask_suffix = post_mask_suffix
        
        # Validation of directory
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Dataset images directory not found: {self.images_dir}")
            
        if self.label_source == "json" and not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Dataset labels directory not found: {self.labels_dir}")
            
        if self.label_source == "mask" and not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Dataset masks directory not found: {self.masks_dir}")
            
        self.samples = self._discover_samples()
        
    def _discover_samples(self):
        samples = []
        for filename in os.listdir(self.images_dir):
            if filename.endswith(self.pre_img_suffix):
                sample_id = filename[:-len(self.pre_img_suffix)]
                
                post_img_path = os.path.join(self.images_dir, sample_id + self.post_img_suffix)
                label_path = os.path.join(self.labels_dir, sample_id + self.post_label_suffix)
                mask_path = os.path.join(self.masks_dir, sample_id + self.post_mask_suffix)
                
                if not os.path.exists(post_img_path):
                    raise FileNotFoundError(f"Missing post-disaster image for sample: {sample_id}")
                    
                if self.label_source == "json" and not os.path.exists(label_path):
                    raise FileNotFoundError(f"Missing JSON annotation for sample: {sample_id}")
                    
                if self.label_source == "mask" and not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Missing mask for sample: {sample_id}")
                    
                samples.append({
                    "sample_id": sample_id,
                    "pre_img_path": os.path.join(self.images_dir, filename),
                    "post_img_path": post_img_path,
                    "label_path": label_path,
                    "mask_path": mask_path
                })
        
        # Deterministic ordering
        samples.sort(key=lambda x: x["sample_id"])
        
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
            
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            pre_image = Image.open(sample["pre_img_path"]).convert("RGB")
            post_image = Image.open(sample["post_img_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load images for {sample['sample_id']}: {str(e)}")
            
        if pre_image.size != post_image.size:
            raise ValueError(f"Pre and post images have inconsistent dimensions for {sample['sample_id']}")
            
        # Generate damage mask
        # PIL size is (W, H), numpy expects (H, W)
        image_shape = (pre_image.size[1], pre_image.size[0])
        
        if self.label_source == "mask":
            mask_np = self._load_mask_file(sample["mask_path"], image_shape)
        else:
            mask_np = self._generate_mask(sample["label_path"], image_shape)
            
        mask_pil = Image.fromarray(mask_np)
        
        if self.is_train:
            pre_tensor, post_tensor, mask_tensor = get_train_transform(pre_image, post_image, mask_pil)
        else:
            pre_tensor, post_tensor, mask_tensor = get_val_transform(pre_image, post_image, mask_pil)
            
        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "damage_mask": mask_tensor,
            "sample_id": sample["sample_id"]
        }

    def _generate_mask(self, label_path, image_shape):
        """
        Parses JSON annotations to create a 2D damage mask.
        image_shape is (H, W)
        """
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Malformed JSON annotation: {label_path}")
            
        # 4 is background
        mask = np.full(image_shape, 4, dtype=np.uint8)
        
        unclassified_target = 4 if self.unclassified_policy == "ignore" else 0
        
        damage_map = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3,
            "un-classified": unclassified_target
        }
        
        features = data.get('features', {}).get('xy', [])
        for feature in features:
            wkt_str = feature.get('wkt')
            if not wkt_str:
                continue
                
            subtype = feature.get('properties', {}).get('subtype', 'no-damage')
            
            if subtype not in damage_map:
                raise ValueError(f"Unsupported damage label '{subtype}' in {label_path}")
                
            try:
                geom = wkt.loads(wkt_str)
            except WKTReadingError:
                raise ValueError(f"Invalid polygon WKT in {label_path}")
                
            if geom.is_empty:
                continue
                
            val = damage_map[subtype]
            
            if geom.geom_type == 'Polygon':
                polys = [geom]
            elif geom.geom_type == 'MultiPolygon':
                polys = list(geom.geoms)
            else:
                continue # ignore other types for now
                
            for polygon in polys:
                # Shapely coordinates are (x, y)
                coords = np.array(polygon.exterior.coords)
                coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(mask, [coords], val)
                
        return mask
        
    def _load_mask_file(self, mask_path, image_shape):
        """
        Loads the pre-generated numeric Kaggle mask, validates values, and remaps to target.
        Kaggle source: 0=bg, 1=no-damage, 2=minor, 3=major, 4=destroyed
        Target: 0=no-damage, 1=minor, 2=major, 3=destroyed, 4=bg
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")
            
        if mask.shape != image_shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_shape}")
            
        unique_vals = np.unique(mask)
        valid_source_vals = {0, 1, 2, 3, 4}
        if not set(unique_vals).issubset(valid_source_vals):
            raise ValueError(f"Unexpected values {unique_vals} found in mask {mask_path}")
            
        remapped_mask = np.full_like(mask, 4)
        remapped_mask[mask == 0] = 4
        remapped_mask[mask == 1] = 0
        remapped_mask[mask == 2] = 1
        remapped_mask[mask == 3] = 2
        remapped_mask[mask == 4] = 3
        
        return remapped_mask
