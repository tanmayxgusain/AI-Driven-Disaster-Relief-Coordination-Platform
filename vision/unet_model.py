"""
Siamese U-Net Model for multi-temporal damage segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    """Basic convolutional block for the U-Net encoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FusionBlock(nn.Module):
    """
    Fuses pre-disaster and post-disaster features at a given scale.
    Concatenates pre, post, and their absolute difference, then applies a convolution.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 3 * in_ch because we concatenate pre, post, and abs(pre - post)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, pre_feat, post_feat):
        diff = torch.abs(pre_feat - post_feat)
        x = torch.cat([pre_feat, post_feat, diff], dim=1)
        return self.fuse(x)

class DecoderBlock(nn.Module):
    """U-Net decoder block with upsampling and skip connection."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d((in_ch // 2) + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle spatial dimensions mismatch if inputs are not perfectly divisible by 2^N
        if x.shape != skip.shape:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
                          
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    """
    Siamese U-Net architecture for multi-temporal satellite imagery.
    
    The model uses a shared encoder for both pre-disaster and post-disaster images,
    computes absolute feature differences at every scale, fuses them, and passes
    them through a standard U-Net decoder.
    
    Expected inputs:
        pre_image: Tensor [B, 3, H, W]
        post_image: Tensor [B, 3, H, W]
        
    Output:
        logits: Tensor [B, 4, H, W] (Classes 0-3 for damage levels)
        
    Background Handling:
        The target mask uses value `4` for background regions. The model DOES NOT
        output a 5th class for background. It outputs 4 classes. During training,
        CrossEntropyLoss(ignore_index=4) must be used so the model only learns to
        classify building damage, ignoring background regions.
    """
    def __init__(self, in_channels=3, num_classes=4, base_channels=32):
        super().__init__()
        
        # Shared Encoder (Weight initialization is handled by PyTorch defaults)
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck (Also shared logically, though we run pre/post through it)
        self.bottleneck = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Fusion Blocks
        self.fuse1 = FusionBlock(base_channels, base_channels)
        self.fuse2 = FusionBlock(base_channels * 2, base_channels * 2)
        self.fuse3 = FusionBlock(base_channels * 4, base_channels * 4)
        self.fuse4 = FusionBlock(base_channels * 8, base_channels * 8)
        self.fuse_bottle = FusionBlock(base_channels * 16, base_channels * 16)
        
        # Decoder
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        
        # Final classifier
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _forward_encoder(self, x):
        """Passes an image through the shared encoder."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottle = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, bottle

    def forward(self, pre_image, post_image):
        if pre_image.shape != post_image.shape:
            raise ValueError(f"Pre and post images must have the same shape. "
                             f"Got {pre_image.shape} and {post_image.shape}.")
                             
        # 1. Feature Extraction (Shared Encoder)
        pre_e1, pre_e2, pre_e3, pre_e4, pre_bottle = self._forward_encoder(pre_image)
        post_e1, post_e2, post_e3, post_e4, post_bottle = self._forward_encoder(post_image)
        
        # 2. Temporal Fusion (Pre + Post + AbsDiff)
        f1 = self.fuse1(pre_e1, post_e1)
        f2 = self.fuse2(pre_e2, post_e2)
        f3 = self.fuse3(pre_e3, post_e3)
        f4 = self.fuse4(pre_e4, post_e4)
        f_bottle = self.fuse_bottle(pre_bottle, post_bottle)
        
        # 3. Decoder
        d4 = self.dec4(f_bottle, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        
        # 4. Final Logits (no softmax applied here)
        logits = self.final_conv(d1)
        return logits
