import os
import argparse
import numpy as np
import traceback
import json
from collections import defaultdict
from vision.xbd_loader import XBDDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Validate xBD dataset readiness")
    parser.add_argument("--dataset-root", required=True, type=str, help="Path to xBD dataset root (e.g. data/xbd)")
    parser.add_argument("--split", required=True, type=str, help="Split to validate (e.g. train, val, tier3)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    return parser.parse_args()

def validate_dataset(args):
    print(f"Validating xBD Dataset in {args.dataset_root}, split: {args.split}")
    
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root {args.dataset_root} not found.")
        return 1
        
    try:
        # Load dataset
        # Set is_train=False to avoid random augmentations during validation
        dataset = XBDDataset(root_dir=args.dataset_root, split=args.split, is_train=False)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return 1
        
    total_samples = len(dataset)
    print(f"Samples discovered: {total_samples}")
    
    if total_samples == 0:
        print("No samples found.")
        return 1
        
    num_to_process = args.max_samples if args.max_samples is not None else total_samples
    num_to_process = min(num_to_process, total_samples)
    print(f"Processing {num_to_process} samples...")
    
    success_count = 0
    broken_count = 0
    
    damage_counts = defaultdict(int)
    instance_counts = defaultdict(int)
    mask_values_seen = set()
    dimensions_seen = set()
    events_seen = defaultdict(int)
    
    for i in range(num_to_process):
        try:
            # We access dataset.samples to extract event id for distribution
            sample_meta = dataset.samples[i]
            sample_id = sample_meta["sample_id"]
            
            # The sample id typically contains the event name, e.g., "hurricane-harvey_00000000"
            if "_" in sample_id:
                event_name = sample_id.split("_")[0]
            else:
                event_name = "unknown"
            events_seen[event_name] += 1
            
            # Read instance counts from the raw JSON
            label_path = sample_meta["label_path"]
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            features = data.get('features', {}).get('xy', [])
            for feature in features:
                if not feature.get('wkt'):
                    continue
                subtype = feature.get('properties', {}).get('subtype', 'no-damage')
                instance_counts[subtype] += 1
            
            # This triggers image loading, mask generation, and transforms
            item = dataset[i]
            
            pre_shape = item["pre_image"].shape
            post_shape = item["post_image"].shape
            mask_shape = item["damage_mask"].shape
            
            if pre_shape != post_shape:
                raise ValueError(f"Pre/Post shape mismatch: {pre_shape} vs {post_shape}")
                
            dimensions_seen.add(tuple(pre_shape))
            
            mask = item["damage_mask"]
            unique_vals = np.unique(mask.numpy())
            mask_values_seen.update(unique_vals)
            
            for v in unique_vals:
                if v == 4:
                    continue # background
                count = (mask == v).sum().item()
                if count > 0:
                    damage_counts[v] += count
                    
            success_count += 1
            
        except Exception as e:
            broken_count += 1
            print(f"Error processing sample {i}: {e}")
            
    print("\n--- Validation Summary ---")
    print(f"Samples successfully loaded: {success_count}")
    print(f"Broken samples: {broken_count}")
    
    print("\nImage Dimensions Seen:")
    for dim in dimensions_seen:
        print(f" - {dim}")
        
    print("\nMask Class Values Present:")
    for v in sorted(list(mask_values_seen)):
        print(f" - {v}")
        
    print("\nDamage Pixel Counts (Mask):")
    print(f" - 0 (no damage): {damage_counts[0]}")
    print(f" - 1 (minor): {damage_counts[1]}")
    print(f" - 2 (major): {damage_counts[2]}")
    print(f" - 3 (destroyed): {damage_counts[3]}")
    
    print("\nBuilding Instance Counts (Raw JSON):")
    print(f" - no-damage: {instance_counts['no-damage']}")
    print(f" - minor-damage: {instance_counts['minor-damage']}")
    print(f" - major-damage: {instance_counts['major-damage']}")
    print(f" - destroyed: {instance_counts['destroyed']}")
    print(f" - un-classified: {instance_counts['un-classified']}")
    
    print("\nEvent/Disaster Distribution (based on sample_id prefix):")
    for event, count in events_seen.items():
        print(f" - {event}: {count} samples")
        
    if broken_count > 0:
        print("\nValidation FAILED due to broken samples.")
        return 1
        
    print("\nValidation PASSED for the processed samples.")
    return 0

if __name__ == "__main__":
    exit(validate_dataset(parse_args()))
