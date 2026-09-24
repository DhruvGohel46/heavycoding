"""
Relativistic Average GAN (RaGAN) Loss for Sen2SR Adversarial Fine-Tuning.
Standard in ESRGAN and Real-ESRGAN: avoids generator collapse with symmetrical gradients.

RaGAN asks: "Is the real image MORE realistic than the generated image on AVERAGE?"
This provides richer gradient signal than vanilla GAN and prevents mode collapse.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativisticGANLoss(nn.Module):
    """
    Relativistic Average GAN (RaGAN) Loss.
    Implements ESRGAN-style Relativistic discriminator loss for both G and D.

    D_loss = -E[log(sigma(D(real) - E[D(fake)]))] - E[log(1 - sigma(D(fake) - E[D(real)]))]
    G_loss = -E[log(sigma(D(fake) - E[D(real)]))] - E[log(1 - sigma(D(real) - E[D(fake)]))]
    """
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, real_logits, fake_logits):
        """
        Compute discriminator loss.
        Args:
            real_logits: D(real_hr) patch map
            fake_logits: D(gen_sr) patch map (detached)
        Returns:
            d_loss: scalar tensor
        """
        mean_fake = fake_logits.mean()
        mean_real = real_logits.mean()

        # Real images should be "more real" than fake on average
        d_real = F.binary_cross_entropy_with_logits(
            real_logits - mean_fake,
            torch.ones_like(real_logits)
        )
        # Fake images should be "less real" than real on average
        d_fake = F.binary_cross_entropy_with_logits(
            fake_logits - mean_real,
            torch.zeros_like(fake_logits)
        )
        return (d_real + d_fake) * 0.5

    def generator_loss(self, real_logits, fake_logits):
        """
        Compute generator loss.
        Args:
            real_logits: D(real_hr) patch map (detached for G step)
            fake_logits: D(gen_sr) patch map
        Returns:
            g_loss: scalar tensor
        """
        mean_fake = fake_logits.mean()
        mean_real = real_logits.mean()

        # Generated images should be "more real" than real on average (flip roles)
        g_fake = F.binary_cross_entropy_with_logits(
            fake_logits - mean_real,
            torch.ones_like(fake_logits)
        )
        # Real images should now be "less convincing"
        g_real = F.binary_cross_entropy_with_logits(
            real_logits - mean_fake,
            torch.zeros_like(real_logits)
        )
        return (g_fake + g_real) * 0.5
