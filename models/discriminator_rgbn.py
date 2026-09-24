"""
4-Band Spectral-Normalized PatchGAN Discriminator for Sen2SR GAN Fine-Tuning.
- Input: 4-channel (B02, B03, B04, B08) HR or SR image
- Architecture: 5-layer conv with Spectral Normalization (Lipschitz constraint)
- Output: Patch-level real/fake discrimination map (70x70 effective receptive field)
- Spectral Norm prevents mode collapse and gradient explosion in 4-band satellite images.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGAN_SN(nn.Module):
    """
    Spectral-Normalized PatchGAN Discriminator.
    Evaluates local 70x70 patches for real vs. fake classification.
    Spectral normalization enforces Lipschitz continuity -> stable GAN training.
    """
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        # Layer 1: No BN on first layer (standard PatchGAN practice)
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4: stride=1 (increasing receptive field without downsampling)
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Output: 1-channel patch-level discrimination map
        self.out = spectral_norm(nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) - either real HR or generated SR tensor
        Returns:
            patch_map: (B, 1, H', W') - patch-level real/fake logits
        """
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return self.out(f4)
