import os
import sys
import json
import pytest
import torch
from PIL import Image

# Ensure the parent directory is in sys.path so 'vision' can be resolved by IDEs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.xbd_loader import XBDDataset

@pytest.fixture
def synthetic_dataset(tmp_path):
    # Create dataset structure
    split_dir = tmp_path / "train"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    masks_dir = split_dir / "masks"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    
    sample_id = "test_001"
    
    # Create pre image (RGB, 1024x1024)
    pre_img = Image.new('RGB', (1024, 1024), color='green')
    pre_img.save(images_dir / f"{sample_id}_pre_disaster.png")
    
    # Create post image
    post_img = Image.new('RGB', (1024, 1024), color='red')
    post_img.save(images_dir / f"{sample_id}_post_disaster.png")
    
    # Create Kaggle-style raw mask (grayscale, 1024x1024)
    # Raw target: 0=bg, 1=no-damage, 2=minor, 3=major, 4=destroyed
    import numpy as np
    mask_np = np.zeros((1024, 1024), dtype=np.uint8)
    mask_np[100:200, 100:200] = 3 # raw major-damage
    mask_img = Image.fromarray(mask_np)
    mask_img.save(masks_dir / f"{sample_id}_post_disaster.png")
    
    # Create annotation
    # A simple square polygon representing a building with major damage
    wkt_poly = "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"
    annotation = {
        "features": {
            "xy": [
                {
                    "properties": {
                        "subtype": "major-damage"
                    },
                    "wkt": wkt_poly
                }
            ]
        }
    }
    
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        json.dump(annotation, f)
        
    return tmp_path

def test_xbd_dataset_initialization(synthetic_dataset):
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False)
    assert len(dataset) == 1
    assert dataset.samples[0]["sample_id"] == "test_001"

def test_missing_files_validation(synthetic_dataset):
    # Remove post image to test validation
    os.remove(synthetic_dataset / "train" / "images" / "test_001_post_disaster.png")
    with pytest.raises(FileNotFoundError, match="Missing post-disaster image"):
        dataset = XBDDataset(synthetic_dataset, split="train")

def test_getitem_shapes_and_types_json(synthetic_dataset):
    # Using val mode (is_train=False) to ensure deterministic evaluation behavior
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="json")
    sample = dataset[0]
    
    pre = sample["pre_image"]
    post = sample["post_image"]
    mask = sample["damage_mask"]
    
    assert pre.shape == (3, 256, 256)
    assert post.shape == (3, 256, 256)
    assert mask.shape == (256, 256)
    
    assert pre.dtype == torch.float32
    assert post.dtype == torch.float32
    assert mask.dtype == torch.int64 # long is int64
    
    # Check that mask values are valid
    unique_vals = torch.unique(mask).numpy()
    assert set(unique_vals).issubset({0, 1, 2, 3, 4})
    
    # Our synthetic poly was major-damage -> class 2
    # Background is class 4
    assert 2 in unique_vals
    assert 4 in unique_vals

def test_missing_annotation_validation(synthetic_dataset):
    os.remove(synthetic_dataset / "train" / "labels" / "test_001_post_disaster.json")
    with pytest.raises(FileNotFoundError, match="Missing JSON annotation"):
        dataset = XBDDataset(synthetic_dataset, split="train", label_source="json")

def test_unknown_damage_label(synthetic_dataset):
    sample_id = "test_001"
    labels_dir = synthetic_dataset / "train" / "labels"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'r') as f:
        data = json.load(f)
    
    data["features"]["xy"][0]["properties"]["subtype"] = "alien-damage"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        json.dump(data, f)
        
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="json")
    with pytest.raises(ValueError, match="Unsupported damage label"):
        _ = dataset[0]

def test_multipolygon_rasterization(synthetic_dataset):
    sample_id = "test_001"
    labels_dir = synthetic_dataset / "train" / "labels"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'r') as f:
        data = json.load(f)
        
    # Valid MultiPolygon
    data["features"]["xy"][0]["wkt"] = "MULTIPOLYGON (((100 100, 200 100, 200 200, 100 200, 100 100)), ((300 300, 400 300, 400 400, 300 400, 300 300)))"
    data["features"]["xy"][0]["properties"]["subtype"] = "destroyed"
    
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        json.dump(data, f)
        
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="json")
    sample = dataset[0]
    mask = sample["damage_mask"]
    
    unique_vals = torch.unique(mask).numpy()
    assert 3 in unique_vals # Destroyed class

def test_malformed_json_annotation(synthetic_dataset):
    sample_id = "test_001"
    labels_dir = synthetic_dataset / "train" / "labels"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        f.write("{ invalid json")
        
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="json")
    with pytest.raises(ValueError, match="Malformed JSON annotation"):
        _ = dataset[0]
def test_unclassified_policy_ignore(synthetic_dataset):
    sample_id = "test_001"
    labels_dir = synthetic_dataset / "train" / "labels"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'r') as f:
        data = json.load(f)
        
    data["features"]["xy"][0]["properties"]["subtype"] = "un-classified"
    
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        json.dump(data, f)
        
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, unclassified_policy="ignore", label_source="json")
    sample = dataset[0]
    mask = sample["damage_mask"]
    
    unique_vals = torch.unique(mask).numpy()
    assert 4 in unique_vals # Background/Ignored
    assert 0 not in unique_vals # Un-classified shouldn't be 0
    assert len(unique_vals) == 1 # Entire mask is 4

def test_unclassified_policy_no_damage(synthetic_dataset):
    sample_id = "test_001"
    labels_dir = synthetic_dataset / "train" / "labels"
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'r') as f:
        data = json.load(f)
        
    data["features"]["xy"][0]["properties"]["subtype"] = "un-classified"
    
    with open(labels_dir / f"{sample_id}_post_disaster.json", 'w') as f:
        json.dump(data, f)
        
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, unclassified_policy="no_damage", label_source="json")
    sample = dataset[0]
    mask = sample["damage_mask"]
    
    unique_vals = torch.unique(mask).numpy()
    assert 0 in unique_vals # Un-classified mapped to 0
    assert 4 in unique_vals # Background

def test_getitem_shapes_and_types_mask(synthetic_dataset):
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="mask")
    sample = dataset[0]
    mask = sample["damage_mask"]
    
    unique_vals = torch.unique(mask).numpy()
    assert set(unique_vals).issubset({0, 1, 2, 3, 4})
    
    # In synthetic_dataset, raw mask has 0 and 3
    # 0 -> 4
    # 3 -> 2
    assert 4 in unique_vals
    assert 2 in unique_vals
    
def test_mask_invalid_values(synthetic_dataset):
    sample_id = "test_001"
    masks_dir = synthetic_dataset / "train" / "masks"
    import numpy as np
    mask_np = np.zeros((1024, 1024), dtype=np.uint8)
    mask_np[100:200, 100:200] = 5 # Invalid!
    mask_img = Image.fromarray(mask_np)
    mask_img.save(masks_dir / f"{sample_id}_post_disaster.png")
    
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, label_source="mask")
    with pytest.raises(ValueError, match="Unexpected values"):
        _ = dataset[0]

def test_max_samples(synthetic_dataset):
    dataset = XBDDataset(synthetic_dataset, split="train", is_train=False, max_samples=0)
    assert len(dataset) == 0

