import os
import torch
import logging
from PIL import Image
from typing import Tuple

from vision.unet_model import SiameseUNet
from vision.preprocessing import get_val_transform

logger = logging.getLogger(__name__)

class CVInferenceService:
    def __init__(self):
        self.model = None
        self.device = None
        self.is_loaded = False

    def load_model(self):
        """Loads the Siamese U-Net from the configured checkpoint path."""
        self.is_loaded = False
        checkpoint_path = os.getenv("CV_CHECKPOINT_PATH")
        device_str = os.getenv("CV_DEVICE", "cpu")
        
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
            
        self.device = torch.device(device_str)
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logger.error(f"CV checkpoint not found at '{checkpoint_path}'. Model unavailable.")
            return False
            
        try:
            self.model = SiameseUNet(in_channels=3, num_classes=4)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # The checkpoint stores the model state dict under 'model_state_dict'
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Loaded CV model successfully on {self.device}.")
            return True
        except Exception as e:
            logger.error(f"Failed to load CV model: {str(e)}")
            return False

    def predict_damage(self, pre_image: Image.Image, post_image: Image.Image) -> Tuple[list, int, int]:
        """
        Runs Siamese U-Net inference on a pre/post image pair.
        Returns:
            damage_map_list: JSON-serializable list of lists representing the 2D mask.
            height: Output height.
            width: Output width.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
            
        if pre_image.size != post_image.size:
            raise ValueError(f"Image dimension mismatch: pre={pre_image.size}, post={post_image.size}")
            
        # Create a dummy mask to reuse the existing validation transform contract safely
        dummy_mask = Image.new('L', pre_image.size, 0)
        
        # Apply validated preprocessing deterministic behavior
        pre_tensor, post_tensor, _ = get_val_transform(pre_image.convert('RGB'), post_image.convert('RGB'), dummy_mask)
        
        # Add batch dimension
        pre_batch = pre_tensor.unsqueeze(0).to(self.device)
        post_batch = post_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(pre_batch, post_batch)
            predicted_mask = torch.argmax(logits, dim=1)
            
        # Remove batch dimension and convert to CPU list
        semantic_map = predicted_mask.squeeze(0).cpu().numpy()
        height, width = semantic_map.shape
        
        return semantic_map.tolist(), height, width

# Global instance for the application lifecycle
cv_service = CVInferenceService()
