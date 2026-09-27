from pydantic import BaseModel
from typing import Dict, List, Optional

class ClassDistributionItem(BaseModel):
    pixels: int
    percentage: float

class DamageAssessmentSchema(BaseModel):
    verification_status: str
    dominant_damage_class: str
    damage_index: float
    total_pixels: int
    affected_pixels: int
    affected_area_ratio: float
    no_damage_pixels: int
    minor_damage_pixels: int
    major_damage_pixels: int
    destroyed_pixels: int
    no_damage_percentage: float
    minor_damage_percentage: float
    major_damage_percentage: float
    destroyed_percentage: float
    affected_percentage: float
    class_distribution: Dict[str, ClassDistributionItem]
from typing import Dict, List

class CVInferenceResponse(BaseModel):
    status: str
    height: int
    width: int
    num_classes: int
    classes: Dict[str, str]
    damage_map: List[List[int]]
    assessment: Optional[DamageAssessmentSchema] = None
