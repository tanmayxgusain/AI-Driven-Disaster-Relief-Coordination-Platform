from typing import List, Dict, Any

class_names = {
    0: "no_damage",
    1: "minor_damage",
    2: "major_damage",
    3: "destroyed"
}

def assess_damage(damage_map: List[List[int]]) -> Dict[str, Any]:
    """
    Computes a project-level damage assessment from a semantic damage map.
    The map must contain only values 0, 1, 2, or 3.
    """
    if not damage_map or not damage_map[0]:
        raise ValueError("Damage map cannot be empty.")
    
    height = len(damage_map)
    width = len(damage_map[0])
    
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_pixels = 0
    
    for row in damage_map:
        if len(row) != width:
            raise ValueError("Damage map must be rectangular.")
        for pixel in row:
            if pixel not in counts:
                raise ValueError(f"Invalid pixel class {pixel}. Must be strictly in {{0, 1, 2, 3}}.")
            counts[pixel] += 1
            total_pixels += 1
            
    affected_pixels = counts[1] + counts[2] + counts[3]
    
    # Percentages
    percentages = {
        0: (counts[0] / total_pixels) * 100.0,
        1: (counts[1] / total_pixels) * 100.0,
        2: (counts[2] / total_pixels) * 100.0,
        3: (counts[3] / total_pixels) * 100.0
    }
    affected_percentage = (affected_pixels / total_pixels) * 100.0
    affected_area_ratio = affected_pixels / total_pixels
    
    # Structural Damage Index
    # A normalized semantic damage score: 0.0 means all pixels are no_damage, 1.0 means all are destroyed.
    # Note: This is a semantic score, not an engineering measurement.
    damage_index = (
        0 * counts[0] + 
        1 * counts[1] + 
        2 * counts[2] + 
        3 * counts[3]
    ) / (3.0 * total_pixels)
    
    # Dominant Damage Class
    # Severity is naturally ordered by the class integer (0 < 1 < 2 < 3)
    # So we sort by count first, then by class ID to break ties by taking the more severe class.
    dominant_class_id = max((counts[c], c) for c in counts)[1]
    dominant_damage_class = class_names[dominant_class_id]
    
    # Verification Status
    # Note: This is a model-derived semantic verification signal, not physical proof of disaster.
    verification_status = "damage_detected" if affected_pixels > 0 else "no_damage_detected"
    
    # Class Distribution Output
    class_distribution = {
        class_names[c]: {
            "pixels": counts[c],
            "percentage": round(percentages[c], 2)
        }
        for c in counts
    }
    
    return {
        "verification_status": verification_status,
        "dominant_damage_class": dominant_damage_class,
        "damage_index": damage_index,
        "total_pixels": total_pixels,
        "affected_pixels": affected_pixels,
        "affected_area_ratio": affected_area_ratio,
        "no_damage_pixels": counts[0],
        "minor_damage_pixels": counts[1],
        "major_damage_pixels": counts[2],
        "destroyed_pixels": counts[3],
        "no_damage_percentage": percentages[0],
        "minor_damage_percentage": percentages[1],
        "major_damage_percentage": percentages[2],
        "destroyed_percentage": percentages[3],
        "affected_percentage": affected_percentage,
        "class_distribution": class_distribution
    }
