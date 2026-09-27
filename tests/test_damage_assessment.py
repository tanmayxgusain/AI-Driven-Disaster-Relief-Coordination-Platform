import pytest
import math
from vision.damage_assessment import assess_damage

def test_all_no_damage():
    # 10x10 map of zeros
    damage_map = [[0 for _ in range(10)] for _ in range(10)]
    result = assess_damage(damage_map)
    
    assert result["damage_index"] == 0.0
    assert result["affected_pixels"] == 0
    assert result["affected_area_ratio"] == 0.0
    assert result["verification_status"] == "no_damage_detected"
    assert result["dominant_damage_class"] == "no_damage"

def test_all_destroyed():
    # 10x10 map of threes
    damage_map = [[3 for _ in range(10)] for _ in range(10)]
    result = assess_damage(damage_map)
    
    assert result["damage_index"] == 1.0
    assert result["affected_area_ratio"] == 1.0
    assert result["verification_status"] == "damage_detected"
    assert result["dominant_damage_class"] == "destroyed"
    assert result["affected_pixels"] == 100

def test_known_distribution():
    # Total pixels = 100
    # 0: 50, 1: 20, 2: 20, 3: 10
    row0 = [0] * 50
    row1 = [1] * 20
    row2 = [2] * 20
    row3 = [3] * 10
    flat_map = row0 + row1 + row2 + row3
    
    # create 10x10
    damage_map = [flat_map[i:i+10] for i in range(0, 100, 10)]
    
    result = assess_damage(damage_map)
    
    assert result["no_damage_pixels"] == 50
    assert result["minor_damage_pixels"] == 20
    assert result["major_damage_pixels"] == 20
    assert result["destroyed_pixels"] == 10
    
    assert result["no_damage_percentage"] == 50.0
    assert result["minor_damage_percentage"] == 20.0
    assert result["major_damage_percentage"] == 20.0
    assert result["destroyed_percentage"] == 10.0
    
    # expected index = (0*50 + 1*20 + 2*20 + 3*10) / 300 = (0 + 20 + 40 + 30) / 300 = 90 / 300 = 0.3
    assert math.isclose(result["damage_index"], 0.3, abs_tol=1e-4)

def test_dominant_class():
    row0 = [0] * 40 # 40
    row1 = [1] * 50 # 50 (highest)
    row2 = [2] * 10 # 10
    flat_map = row0 + row1 + row2
    damage_map = [flat_map[i:i+10] for i in range(0, 100, 10)]
    
    result = assess_damage(damage_map)
    assert result["dominant_damage_class"] == "minor_damage"

def test_tie_handling():
    # Tie between minor (1) and major (2)
    # Both have 40
    row0 = [0] * 20
    row1 = [1] * 40
    row2 = [2] * 40
    flat_map = row0 + row1 + row2
    damage_map = [flat_map[i:i+10] for i in range(0, 100, 10)]
    
    result = assess_damage(damage_map)
    assert result["dominant_damage_class"] == "major_damage"

def test_invalid_class():
    damage_map = [[0, 1], [2, 4]]
    with pytest.raises(ValueError, match="Invalid pixel class 4"):
        assess_damage(damage_map)
        
    damage_map2 = [[-1, 1], [2, 3]]
    with pytest.raises(ValueError, match="Invalid pixel class -1"):
        assess_damage(damage_map2)

def test_empty_input():
    with pytest.raises(ValueError, match="Damage map cannot be empty"):
        assess_damage([])
        
    with pytest.raises(ValueError, match="Damage map cannot be empty"):
        assess_damage([[]])

def test_percentage_consistency():
    row0 = [0] * 33
    row1 = [1] * 33
    row2 = [2] * 33
    row3 = [3] * 1
    flat_map = row0 + row1 + row2 + row3
    damage_map = [flat_map[i:i+10] for i in range(0, 100, 10)]
    
    result = assess_damage(damage_map)
    
    total_pct = (result["no_damage_percentage"] + 
                 result["minor_damage_percentage"] + 
                 result["major_damage_percentage"] + 
                 result["destroyed_percentage"])
                 
    assert math.isclose(total_pct, 100.0, abs_tol=1e-2)

def test_api_integration():
    from main import app
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    
    client = TestClient(app)
    
    # Provide deterministic mock for cv_service.predict_damage
    # returning a solid '3' map
    synthetic_map = [[3 for _ in range(256)] for _ in range(256)]
    
    with patch("backend.routers.cv.cv_service.is_loaded", True), \
         patch("backend.routers.cv.cv_service.predict_damage", return_value=(synthetic_map, 256, 256)):
        
        # We also need dummy file bytes
        import io
        from PIL import Image
        
        img = Image.new("RGB", (256, 256), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        dummy_bytes = buf.getvalue()
        
        response = client.post(
            "/api/v1/cv/infer",
            files={
                "pre_event_image": ("pre.png", dummy_bytes, "image/png"),
                "post_event_image": ("post.png", dummy_bytes, "image/png")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "assessment" in data
        assessment = data["assessment"]
        assert assessment["verification_status"] == "damage_detected"
        assert assessment["dominant_damage_class"] == "destroyed"
        assert assessment["damage_index"] == 1.0
