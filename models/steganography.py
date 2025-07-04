import torch
import torch.nn as nn
from .inn import InvertibleSteganoNetwork

class SteganoNetwork(nn.Module):
    def __init__(self, in_channels=3, num_blocks=8):
        super().__init__()
        self.inn = InvertibleSteganoNetwork(in_channels=in_channels, num_blocks=num_blocks)

    def forward(self, image, template=None, extract_mode=False):
        if extract_mode:
            # This is not how extraction works with INNs.
            # We need the stego_image and the stego_template_remains
            # This method signature might need a rethink, as `extract_template` is the proper way.
            # For now, let's assume `extract_template` is called directly.
            raise NotImplementedError("Use extract_template for INNs.")
        else:
            # Normal forward mode - embed template into image
            stego_image, stego_template_remains = self.inn.forward(image, template)
            return stego_image, stego_template_remains

    def extract_template(self, stego_image, stego_template_remains):
        """Extract the hidden template from a steganographic image."""
        _, extracted_template = self.inn.reverse(stego_image, stego_template_remains)
        return extracted_template
        
    def reverse(self, stego_image, stego_template_remains):
        """Reverses the steganography process to get the original image and template."""
        return self.inn.reverse(stego_image, stego_template_remains)

class SteganoLoss(nn.Module):
    def __init__(self, lambda_cover=1.0, lambda_secret=1.0, lambda_remains=1.0):
        super().__init__()
        self.lambda_cover = lambda_cover
        self.lambda_secret = lambda_secret
        self.lambda_remains = lambda_remains
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, original_image, stego_image, original_template, extracted_template, stego_template_remains):
        
        # Loss for the cover image (stego vs original)
        loss_cover = self.mse_loss(stego_image, original_image)
        
        # Loss for the secret message (extracted vs original)
        loss_secret = self.mse_loss(extracted_template, original_template)
        
        # Loss for the remainder, should be close to zero
        loss_remains = self.mse_loss(stego_template_remains, torch.zeros_like(stego_template_remains))
        
        total_loss = (self.lambda_cover * loss_cover + 
                      self.lambda_secret * loss_secret + 
                      self.lambda_remains * loss_remains)
        
        return total_loss, loss_cover, loss_secret, loss_remains 