import io
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image, UnidentifiedImageError
from backend.schemas import CVInferenceResponse
from backend.services.cv_inference import cv_service
from vision.damage_assessment import assess_damage

router = APIRouter(prefix="/api/v1/cv", tags=["Computer Vision"])

@router.get("/health")
def cv_health_check():
    return {
        "service": "cv",
        "model_loaded": cv_service.is_loaded,
        "device": str(cv_service.device) if cv_service.device else "uninitialized"
    }

@router.post("/infer", response_model=CVInferenceResponse)
async def infer_damage(
    pre_event_image: UploadFile = File(...),
    post_event_image: UploadFile = File(...)
):
    if not cv_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CV Inference service is not available (checkpoint missing or failed to load)."
        )
        
    # Read and decode images
    try:
        pre_bytes = await pre_event_image.read()
        post_bytes = await post_event_image.read()
        
        pre_img = Image.open(io.BytesIO(pre_bytes))
        post_img = Image.open(io.BytesIO(post_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded files must be valid image formats (e.g., PNG, JPEG)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded files: {str(e)}"
        )
        
    if pre_img.size != post_img.size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Mismatched image dimensions: pre_event={pre_img.size}, post_event={post_img.size}."
        )
        
    try:
        damage_map, height, width = cv_service.predict_damage(pre_img, post_img)
        assessment = assess_damage(damage_map)
        
        return CVInferenceResponse(
            status="success",
            height=height,
            width=width,
            num_classes=4,
            classes={
                "0": "no_damage",
                "1": "minor_damage",
                "2": "major_damage",
                "3": "destroyed"
            },
            damage_map=damage_map,
            assessment=assessment
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal model inference failure."
        )
