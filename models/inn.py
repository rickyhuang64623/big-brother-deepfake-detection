import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT_LL(nn.Module):
    """
    Extract only the LL (low-low frequency) sub-band using Haar wavelet transform.
    Used specifically for L_sec loss calculation (equation 5).
    """
    def __init__(self):
        super().__init__()
        # Haar wavelet coefficients for LL sub-band (broadcastable to any channel count)
        self.register_buffer('ll_weights', torch.tensor([1., 1., 1., 1.]).view(1, 1, 4, 1, 1) / 4.0)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Ensure even dimensions
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            B, C, H, W = x.shape
            
        # Unfold input to patches
        x = x.unfold(2, 2, 2).unfold(3, 2, 2)  # [B, C, H//2, W//2, 2, 2]
        x = x.contiguous().view(B, C, H//2, W//2, 4)
        x = x.permute(0, 1, 4, 2, 3)  # [B, C, 4, H//2, W//2]
        
        # Apply only LL transform (broadcast weights to all channels)
        return (x * self.ll_weights).sum(dim=2)

class AffineCouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.split_size = in_channels // 2
        
        self.subnet = nn.Sequential(
            nn.Conv2d(self.split_size, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, self.split_size * 2, 3, padding=1)
        )
        
        # Initialize the last layer to be zero
        with torch.no_grad():
            self.subnet[-1].weight.zero_()
            self.subnet[-1].bias.zero_()

    def forward(self, x, reverse=False):
        x1, x2 = torch.split(x, self.split_size, dim=1)
        
        if not reverse:
            s_t = self.subnet(x1.clone())
            s, t = torch.split(s_t, self.split_size, dim=1)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
        else:
            s_t = self.subnet(x1.clone())
            s, t = torch.split(s_t, self.split_size, dim=1)
            y2 = (x2 - t) / torch.exp(s)
            y = torch.cat([x1, y2], dim=1)
            
        return y

class InvertibleSteganoNetwork(nn.Module):
    def __init__(self, num_blocks=8, in_channels=3):
        super().__init__()
        self.blocks = nn.ModuleList([AffineCouplingBlock(in_channels * 2) for _ in range(num_blocks)])
        
        # DWT_LL module for L_sec loss calculation (equation 5)
        self.dwt_ll = DWT_LL()

    def forward(self, image, template):
        x = torch.cat([image, template], dim=1)
        
        for block in self.blocks:
            x = block(x)
            x = x.roll(shifts=1, dims=1) # Permutation
            
        stego_image, stego_template_remains = torch.split(x, 3, dim=1)
        return stego_image, stego_template_remains

    def reverse(self, stego_image, stego_template_remains):
        x = torch.cat([stego_image, stego_template_remains], dim=1)
        
        for block in reversed(self.blocks):
            x = x.roll(shifts=-1, dims=1) # Inverse Permutation
            x = block(x, reverse=True)
            
        image, template = torch.split(x, 3, dim=1)
        return image, template
    
    def calculate_l_sec_loss(self, image, template):
        """
        Calculate L_sec loss (equation 5): ||DWT_LL(S) - DWT_LL(I)||_2
        
        Args:
            image: Input image I [B, C, H, W]
            template: Secret template S [B, C, H, W]
            
        Returns:
            L_sec loss value
        """
        # Extract LL sub-band of the image
        dwt_ll_image = self.dwt_ll(image)
        
        # Extract LL sub-band of the template
        dwt_ll_template = self.dwt_ll(template)
        
        # Calculate L2 norm between the two LL sub-bands
        l_sec_loss = F.mse_loss(dwt_ll_template, dwt_ll_image)
        
        return l_sec_loss 