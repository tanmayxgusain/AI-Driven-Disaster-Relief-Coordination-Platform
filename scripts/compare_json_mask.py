import os
import argparse
import numpy as np
import cv2
import json
from shapely import wkt

def compare_masks(mask_path, label_path):
    # Load JSON mask
    with open(label_path, 'r') as f:
        data = json.load(f)
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return
        
    image_shape = mask.shape
    json_mask = np.full(image_shape, 4, dtype=np.uint8)
    
    damage_map = {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
        "un-classified": 4
    }
    
    features = data.get('features', {}).get('xy', [])
    for feature in features:
        wkt_str = feature.get('wkt')
        if not wkt_str: continue
        subtype = feature.get('properties', {}).get('subtype', 'no-damage')
        geom = wkt.loads(wkt_str)
        if geom.is_empty: continue
        
        val = damage_map.get(subtype, 0)
        
        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
        for polygon in polys:
            coords = np.array(polygon.exterior.coords)
            coords = np.round(coords).astype(np.int32)
            cv2.fillPoly(json_mask, [coords], val)
            
    # Remap Kaggle Mask to target values
    remapped_mask = np.full_like(mask, 4)
    remapped_mask[mask == 0] = 4
    remapped_mask[mask == 1] = 0
    remapped_mask[mask == 2] = 1
    remapped_mask[mask == 3] = 2
    remapped_mask[mask == 4] = 3
    
    # Compare
    match = (json_mask == remapped_mask)
    pixel_agreement = match.sum() / match.size
    print(f"\nComparing: {os.path.basename(mask_path)}")
    print(f"Pixel Agreement: {pixel_agreement:.4f}")
    
    print(f"JSON mask unique values: {np.unique(json_mask)}")
    print(f"Provided mask unique values (remapped): {np.unique(remapped_mask)}")
    
    if pixel_agreement < 0.99:
        print("Major discrepancies found! (Agreement < 99%)")
    else:
        print("Masks are highly consistent.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic utility to compare JSON vs provided numeric masks.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="tier1")
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()
    
    masks_dir = os.path.join(args.dataset_root, args.split, "masks")
    labels_dir = os.path.join(args.dataset_root, args.split, "labels")
    
    if not os.path.exists(masks_dir) or not os.path.exists(labels_dir):
        print("Error: Masks or Labels directory not found.")
        exit(1)
        
    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_post_disaster.png"))
    for mf in mask_files[:args.max_samples]:
        sample_id = mf.replace("_post_disaster.png", "")
        mask_path = os.path.join(masks_dir, mf)
        label_path = os.path.join(labels_dir, sample_id + "_post_disaster.json")
        if os.path.exists(label_path):
            compare_masks(mask_path, label_path)
