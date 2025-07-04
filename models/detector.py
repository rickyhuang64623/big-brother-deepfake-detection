import torch
import torch.nn as nn
import torch.nn.functional as F

class Detector(nn.Module):
    def __init__(self, in_channels=3, template_size=128):
        super().__init__()
        self.template_size = template_size
        
        # Template extraction network - extracts the hidden template from images
        self.template_extractor = nn.Sequential(
            # First block
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            
            # Second block
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            
            # Third block
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            
            # Fourth block
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        
        # Template reconstruction head - reconstructs the hidden template
        self.template_reconstructor = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()  # Output template in [-1, 1] range
        )
        
        # Template integrity analyzer - analyzes if the extracted template is intact
        self.integrity_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features from the image
        features = self.template_extractor(x)
        
        # Reconstruct the hidden template
        extracted_template = self.template_reconstructor(features)
        
        # Analyze template integrity to determine if it's real (1.0) or fake (0.0)
        integrity_score = self.integrity_analyzer(features)
        
        return integrity_score, extracted_template

class DetectorLoss(nn.Module):
    def __init__(self, lambda_integrity=1.0, lambda_template=0.1):
        super().__init__()
        self.lambda_integrity = lambda_integrity
        self.lambda_template = lambda_template
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_integrity, pred_template, target_integrity, target_template=None):
        # Integrity loss (real vs fake classification)
        integrity_loss = self.bce_loss(pred_integrity, target_integrity)
        
        # Template reconstruction loss (if target template is provided)
        if target_template is not None:
            template_loss = self.mse_loss(pred_template, target_template)
            total_loss = self.lambda_integrity * integrity_loss + self.lambda_template * template_loss
        else:
            template_loss = torch.tensor(0.0).to(pred_integrity.device)
            total_loss = integrity_loss
        
        return total_loss, integrity_loss, template_loss 